import streamlit as st
import json
import os
import base64
from tempfile import NamedTemporaryFile
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Optional, Dict


# Initialize the OpenAI client
client = OpenAI(api_key="your-openai-api-key")  

# For the vision model, you need either:
# Option 1: If using OpenAI's Vision model
# (already included in the OpenAI package)

# Option 2: If using another vision model like Google's Vertex AI
# pip install google-cloud-aiplatform
# from vertexai.vision_models import ImageCaptioningModel
# vision_model = ImageCaptioningModel.from_pretrained("your-model-name")

# Define the schema for product search parameters
class AmazonSearchQuery(BaseModel):
    product_name: str = Field(..., description="Main product name extracted from the image")
    brand: Optional[str] = Field(None, description="Brand name if visible in the image")
    product_size: Optional[str] = Field(None, description="Size/amount/weight information (e.g., '16oz', '500ml')")
    identifiers: Optional[Dict[str, str]] = Field(None, description="Product identifiers such as UPC, EAN, ASIN if visible")
    key_features: Optional[List[str]] = Field(None, description="Key product features or descriptors visible in the image")
    
# Define the schema for Amazon search results verification
class ProductVerification(BaseModel):
    match_found: bool = Field(..., description="Whether a matching product was found on Amazon")
    confidence_score: float = Field(..., description="Confidence score for the match (0-1)")
    matched_product: Optional[Dict] = Field(None, description="Details of the matched product")
    discrepancies: Optional[List[str]] = Field(None, description="List of discrepancies between image and found product")

def amazon_search(search_parameters, rainforest_api_key):
    """
    Search Amazon for products matching parameters extracted from a product image.
    
    Input: Structured data from VLM image analysis
    Output: Top matching Amazon products with detailed information
    """
    import requests
    
    # Use the provided API key
    api_key = rainforest_api_key
    base_url = "https://api.rainforestapi.com/request"
    
    # Configure request parameters
    params = {
        "api_key": api_key,
        "amazon_domain": "amazon.com"
    }
    
    # Determine search approach based on available information
    if "asin" in search_parameters and search_parameters["asin"]:
        # Direct ASIN lookup if available
        params["type"] = "product"
        params["asin"] = search_parameters["asin"]
    elif "upc" in search_parameters or "ean" in search_parameters or "isbn" in search_parameters:
        # GTIN/UPC/EAN lookup
        params["type"] = "product"
        params["gtin"] = search_parameters.get("upc") or search_parameters.get("ean") or search_parameters.get("isbn")
    else:
        # Text-based search using extracted information
        params["type"] = "search"
        
        # Construct search term from product name and attributes
        search_term = search_parameters["product_name"]
        if "brand" in search_parameters:
            search_term = f"{search_parameters['brand']} {search_term}"
        if "attributes" in search_parameters:
            for key, value in search_parameters["attributes"].items():
                search_term += f" {value}"
        
        params["search_term"] = search_term
    
    # Make API request
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise exception for HTTP errors
        data = response.json()
        
        # Process results based on request type
        results = []
        if params["type"] == "product":
            if "product" in data:
                product = data["product"]
                results.append({
                    "asin": product.get("asin"),
                    "title": product.get("title"),
                    "brand": product.get("brand"),
                    "price": product.get("buybox_winner", {}).get("price"),
                    "rating": product.get("rating"),
                    "ratings_total": product.get("ratings_total"),
                    "main_image": product.get("main_image", {}).get("link"),
                    "link": product.get("link"),
                    "dimensions": product.get("dimensions"),
                    "weight": product.get("weight")
                })
        elif params["type"] == "search":
            for item in data.get("search_results", [])[:5]:  # Limit to top 5 results
                results.append({
                    "asin": item.get("asin"),
                    "title": item.get("title"),
                    "link": item.get("link"),
                    "image": item.get("image"),
                    "price": item.get("price")
                })
        
        return {
            "query_info": search_parameters,
            "results": results
        }
    except Exception as e:
        # Handle any errors
        return {
            "query_info": search_parameters,
            "error": str(e),
            "results": []
        }
    
def extract_product_details_from_image(image_path, client, model_id="accounts/fireworks/models/llama-v3p1-8b-instruct", timing=None):
    """
    Extract product details from an image using document inlining and a specified model.
    
    Parameters:
    image_path (str): Path to the product image file
    client: API client (OpenAI or compatible)
    model_id (str): Model ID to use for extraction
    timing (dict, optional): Timing dictionary to update with performance metrics
    
    Returns:
    dict: Extracted product details
    """
    import base64
    import time
    import json
    
    # Track timing if requested
    if timing is not None:
        image_prep_start = time.time()
    
    # Prepare image for document inlining
    with open(image_path, "rb") as image_file:
        image_content = image_file.read()
        
    base64_image = base64.b64encode(image_content).decode('utf-8')
    
    # Determine MIME type based on file extension
    if image_path.lower().endswith('.jpg') or image_path.lower().endswith('.jpeg'):
        mime_type = "image/jpeg"
    elif image_path.lower().endswith('.png'):
        mime_type = "image/png"
    else:
        mime_type = "image/jpeg"  # Default to JPEG
    
    # Create image URL with document inlining transform
    image_url = f"data:{mime_type};base64,{base64_image}#transform=inline"
    
    if timing is not None:
        timing['image_preparation'] = time.time() - image_prep_start
        extract_start = time.time()
    
    # Extract product details from image using document inlining
    extract_messages = [
        {"role": "system", "content": "Extract detailed product information from the image. Include product name, brand, size/weight, and any visible identifiers like UPC or model numbers."},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": "What product is shown in this image? Extract all visible details including product name, brand, size/weight, and any identifiers."}
        ]}
    ]
    
    # Use document inlining to extract product details with JSON output
    extract_response = client.chat.completions.create(
        model=model_id,
        messages=extract_messages,
        response_format={"type": "json_object", "schema": AmazonSearchQuery.model_json_schema()},
        max_tokens=1000
    )
    
    # Parse extracted product details
    content = extract_response.choices[0].message.content
    if content.startswith("<think>"):
        content = content.split("</think>", 1)[1].strip()
    product_details = json.loads(content)
    
    if timing is not None:
        timing['product_extraction'] = time.time() - extract_start
    
    return product_details

def extract_product_details_from_image_firesearch(image_path, client, timing=None):
    """
    Extract product details from an image using the firesearch-ocr-v6 model.
    
    Parameters:
    image_path (str): Path to the product image file
    client: API client (OpenAI compatible)
    timing (dict, optional): Timing dictionary to update with performance metrics
    
    Returns:
    dict: Extracted product details
    """
    import base64
    import time
    import json
    
    # Track timing if requested
    if timing is not None:
        image_prep_start = time.time()
    
    # Encode image to base64
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Determine MIME type based on file extension
    if image_path.lower().endswith('.jpg') or image_path.lower().endswith('.jpeg'):
        mime_type = "image/jpeg"
    elif image_path.lower().endswith('.png'):
        mime_type = "image/png"
    else:
        mime_type = "image/jpeg"  # Default to JPEG
        
    if timing is not None:
        timing['image_preparation'] = time.time() - image_prep_start
        extract_start = time.time()
    
    # Extract product details using firesearch-ocr-v6 model
    extract_messages = [
        {"role": "system", "content": "You are an OCR and product information extraction assistant. Extract detailed product information from the image. Include product name, brand, size/weight, and any visible identifiers like UPC, EAN, or model numbers."},
        {"role": "user", "content": [
            {"type": "text", "text": "Extract all product information visible in this image. Include the product name, brand, size/weight, and any identifiers like UPC, EAN, or model numbers. Format your response as JSON."},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
        ]}
    ]
    
    # Make API call to firesearch-ocr model
    extract_response = client.chat.completions.create(
        model="accounts/fireworks/models/phi-3-vision-128k-instruct",
        messages=extract_messages,
        response_format={"type": "json_object", "schema": AmazonSearchQuery.model_json_schema()},
        max_tokens=1000
    )
    
    # Parse extracted product details
    product_details = json.loads(extract_response.choices[0].message.content)
    
    if timing is not None:
        timing['product_extraction'] = time.time() - extract_start
    
    return product_details

def process_product_image(uploaded_image, fireworks_api_key, rainforest_api_key):
    """Modified version to work with Streamlit file uploads"""
    import time
    import json
    
    # Track timing information
    timing = {}
    start_total = time.time()
    
    # Create a placeholder for current status updates
    status_placeholder = st.empty()
    
    # Initialize sections for collapsible results
    image_analysis_section = st.container()
    amazon_search_section = st.container()
    verification_section = st.container()
    
    status_placeholder.write("🔍 Processing uploaded image...")
    
    # Save the uploaded file to a temporary file
    with NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
        tmp_file.write(uploaded_image.getbuffer())
        image_path = tmp_file.name
    
    # Initialize client with API key from secrets
    client_start = time.time()
    client = OpenAI(
        base_url="https://api.fireworks.ai/inference/v1",
        api_key=fireworks_api_key
    )
    timing['client_initialization'] = time.time() - client_start
    
    # Extract product details from the image
    extract_start = time.time()
    status_placeholder.write("📷 Analyzing image with AI... ")
    product_details = extract_product_details_from_image(
        image_path=image_path,
        client=client,
        timing=timing
    )
    extract_time = time.time() - extract_start
    
    # Clear the status placeholder
    status_placeholder.empty()
    
    # Write permanent completion message
    with image_analysis_section:
        st.write(f"✅ Image analyzed in {extract_time:.2f}s")
        with st.expander("Image Analysis Details"):
            st.json(product_details)
    
    # Execute Amazon search using Rainforest API
    search_start = time.time()
    status_placeholder.write(f"🔎 Searching Amazon for matching products...")
    amazon_results = amazon_search(product_details, rainforest_api_key)
    search_time = time.time() - search_start
    timing['amazon_search'] = search_time
    
    # Clear the status placeholder
    status_placeholder.empty()
    
    # Write permanent completion message
    with amazon_search_section:
        st.write(f"✅ Amazon search completed in {search_time:.2f}s")
        with st.expander("Amazon Search Results"):
            st.json(amazon_results)
    
    # Check if we have results before proceeding
    if not amazon_results.get("results") or len(amazon_results["results"]) == 0:
        st.error("❌ No products found in Amazon search")
        return {
            "match_found": False,
            "top_match": None,
            "discrepancies": [],
            "timing": timing,
            "total_time": time.time() - start_total
        }
    
    # Verify product match with LLM using JSON mode
    verify_start = time.time()
    status_placeholder.write(f"⚖️ Verifying matches...")
    verification_messages = [
        {"role": "system", "content": "Analyze the Amazon product results and determine how well they match the original product details. Assign a confidence score (0-1) to each result based on how closely it matches."},
        {"role": "user", "content": f"Compare these product details:\n\nOriginal product: {json.dumps(product_details)}\n\nAmazon results: {json.dumps(amazon_results)}\n\nFor each result, provide a confidence score and list any discrepancies. DO NOT include links or URLs in your response. Format your response as a JSON object with 'ranked_matches' (array of objects with 'asin', 'confidence_score', and 'discrepancies' fields) and 'general_discrepancies' (array of strings)."}
    ]
    
    verification_response = client.chat.completions.create(
        model="accounts/fireworks/models/llama-v3p1-8b-instruct",
        messages=verification_messages,
        response_format={"type": "json_object"},
        max_tokens=1000
    )
    
    # Parse verification results with error handling
    try:
        llm_verification = json.loads(verification_response.choices[0].message.content)
        verification_time = time.time() - verify_start
        
        # Clear the status placeholder
        status_placeholder.empty()
        
        # Write permanent completion message
        with verification_section:
            st.write(f"✅ Verification completed in {verification_time:.2f}s")
            with st.expander("Verification Results"):
                st.json(llm_verification)
                
        timing['verification'] = verification_time
                
    except json.JSONDecodeError as e:
        # Clear the status placeholder
        status_placeholder.empty()
        
        # Write permanent error message
        with verification_section:
            st.error(f"❌ Error parsing verification response: {str(e)}")
            with st.expander("Verification Error"):
                st.write(f"Raw response: {verification_response.choices[0].message.content[:500]}... (truncated)")
        
        # Create a default fallback structure
        llm_verification = {
            "ranked_matches": [
                {"asin": result.get("asin"), "confidence_score": 0.5, "discrepancies": []}
                for result in amazon_results.get("results", [])
            ],
            "general_discrepancies": []
        }
        verification_time = time.time() - verify_start
        timing['verification'] = verification_time
    
    # Process and enhance the results with direct information from the Amazon API
    processing_start = time.time()
    status_placeholder.write(f"🏁 Processing final results...")
    
    # Create the final output structure
    final_results = {
        "match_found": False,
        "top_match": None,
        "discrepancies": [],
        "timing": timing
    }
    
    # Process matches
    matches = []
    for result in amazon_results.get("results", []):
        # Get or estimate the confidence score
        confidence = 0.0
        discrepancies = []
        
        # Try to find this result in the LLM verification
        if "ranked_matches" in llm_verification:
            for match in llm_verification["ranked_matches"]:
                if match.get("asin") == result.get("asin"):
                    confidence = match.get("confidence_score", 0.0)
                    discrepancies = match.get("discrepancies", [])
                    break
        
        matches.append({
            "confidence_score": confidence,
            "product_data": result,
            "discrepancies": discrepancies
        })
    
    # Sort matches by confidence score in descending order
    matches.sort(key=lambda x: x["confidence_score"], reverse=True)
    
    # Set match_found if we have at least one match
    if matches:
        final_results["match_found"] = True
        
        # Add only the top match to final results
        top_match = matches[0]
        
        # Take the product data directly from the Amazon API results
        product_data = top_match["product_data"]
        
        # Build match info using only data directly from the API
        match_info = {
            "title": product_data.get("title", ""),
            "asin": product_data.get("asin", ""),
            "price": product_data.get("price", ""),
            "discrepancies": top_match["discrepancies"],
            # Use the link directly from the API response, not from verification
            "link": product_data.get("link", ""),
            "image": product_data.get("main_image", product_data.get("image", ""))
        }
        
        # Add additional fields if available
        for field in ["brand", "rating", "ratings_total", "dimensions", "weight"]:
            if field in product_data and product_data[field]:
                match_info[field] = product_data[field]
                
        final_results["top_match"] = match_info
        
    # Add overall discrepancies from LLM
    if "general_discrepancies" in llm_verification:
        final_results["discrepancies"] = llm_verification["general_discrepancies"]
    
    timing['results_processing'] = time.time() - processing_start
    timing['total_time'] = time.time() - start_total
    # Calculate Fireworks processing time (total time minus Amazon search time)
    timing['fireworks_processing'] = timing['total_time'] - timing.get('amazon_search', 0)
    
    # Clear the status placeholder
    status_placeholder.empty()
    
    # Print final match result
    st.divider()  # Add a visual separator for the final results
    
    if final_results["top_match"]:
        st.success("✨ Found a match on Amazon!")
        
        # Create columns for product details
        col1, col2 = st.columns([1, 2])
        
        # Show product image in first column
        if 'image' in final_results['top_match']:
            image_url = final_results['top_match']['image']
            if isinstance(image_url, dict) and 'link' in image_url:
                image_url = image_url['link']
            col1.image(image_url, width=200, use_container_width=False)
        
        # Show details in second column
        col2.write(f"**Title:** {final_results['top_match']['title']}")
        if 'brand' in final_results['top_match']:
            col2.write(f"**Brand:** {final_results['top_match']['brand']}")
        
        # Handle price display correctly
        price = final_results['top_match']['price']
        if isinstance(price, dict) and 'raw' in price:
            col2.write(f"**Price:** {price['raw']}")
        elif price:
            col2.write(f"**Price:** {price}")
            
        # Print link only if it exists
        if final_results['top_match']['link']:
            col2.markdown(f"[View on Amazon]({final_results['top_match']['link']})")
            
        # Discrepancies
        if final_results.get('discrepancies'):
            st.subheader("Potential discrepancies:")
            for disc in final_results['discrepancies']:
                st.write(f"- {disc}")
    else:
        st.error("❌ No suitable match found on Amazon")
    
    # Print overall timing
    st.write(f"⏱️ **Total processing time:** {timing['total_time']:.2f} seconds")
    st.write(f"🪄 **Fireworks processing time:** {timing['fireworks_processing']:.2f} seconds")
    
    # Overall processing details (in collapsed section)
    with st.expander("Processing details"):
        st.json(final_results)
    
    # Clean up the temporary file
    os.unlink(image_path)
    
    return final_results

# Streamlit UI
st.title("Product Image Analyzer")
st.write("Upload a product image to search for it on Amazon")

# API Key configuration section
st.sidebar.header("API Configuration")

# Try to get API keys from secrets first (for developer's deployment)
try:
    default_fireworks_key = st.secrets["fireworks"]["api_key"]
    default_rainforest_key = st.secrets["rainforest"]["api_key"]
    st.sidebar.success("✅ API keys loaded from configuration")
except:
    default_fireworks_key = ""
    default_rainforest_key = ""
    st.sidebar.write("This app requires two API keys to function:")

# Fireworks API key input
fireworks_api_key = st.sidebar.text_input(
    "Fireworks AI API Key", 
    value=default_fireworks_key,
    type="password",
    help="Get your API key from https://fireworks.ai"
)

# Rainforest API key input  
rainforest_api_key = st.sidebar.text_input(
    "Rainforest API Key", 
    value=default_rainforest_key,
    type="password",
    help="Get your API key from https://rainforestapi.com"
)

# Check if API keys are provided
if not fireworks_api_key or not rainforest_api_key:
    st.warning("⚠️ Please enter both API keys in the sidebar to use this app.")
    st.info("""
    **To get started:**
    1. Get a Fireworks AI API key from [fireworks.ai](https://fireworks.ai)
    2. Get a Rainforest API key from [rainforestapi.com](https://rainforestapi.com)
    3. Enter both keys in the sidebar
    4. Upload a product image to analyze
    """)

# File uploader
uploaded_file = st.file_uploader("Choose a product image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    
    # Process button
    if st.button("Analyze Product"):
        if not fireworks_api_key or not rainforest_api_key:
            st.error("Please provide both API keys in the sidebar first.")
        else:
            with st.spinner(""):  # Empty spinner as we'll use our own status updates
                results = process_product_image(uploaded_file, fireworks_api_key, rainforest_api_key)
else:
    st.info("Please upload an image to begin analysis")

