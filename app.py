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

def amazon_search(search_parameters):
    """
    Search Amazon for products matching parameters extracted from a product image.
    
    Input: Structured data from VLM image analysis
    Output: Top matching Amazon products with detailed information
    """
    import requests
    
    api_key = "FB0C32E1DCB3433C997C0A0FB1E70608"
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
    
def extract_product_details_from_image_mixtral(image_path, client, model_id="accounts/fireworks/models/mixtral-8x7b-instruct", timing=None):
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
    product_details = json.loads(extract_response.choices[0].message.content)
    
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

def process_product_image(uploaded_image, api_key):
    """Modified version to work with Streamlit file uploads"""
    import time
    import json
    
    # Track timing information
    timing = {}
    start_total = time.time()
    
    st.write("🔍 Processing uploaded image...")
    
    # Save the uploaded file to a temporary file
    with NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
        tmp_file.write(uploaded_image.getbuffer())
        image_path = tmp_file.name
    
    # Initialize client
    client_start = time.time()
    client = OpenAI(
        base_url="https://api.fireworks.ai/inference/v1",
        api_key=api_key
    )
    timing['client_initialization'] = time.time() - client_start
    
    # Extract product details from the image
    extract_start = time.time()
    st.write("📷 Analyzing image with AI... ")
    product_details = extract_product_details_from_image_mixtral(
        image_path=image_path,
        client=client,
        timing=timing
    )
    extract_time = time.time() - extract_start
    
    # Display extracted information
    print(f"✅ Image analyzed in {extract_time:.2f}s")
    print(f"   Product: {product_details.get('brand', 'Unknown brand')} {product_details.get('product_name', 'Unknown product')}")
    if product_details.get('product_size'):
        print(f"   Size: {product_details.get('product_size')}")
    
    # Execute Amazon search using Rainforest API
    search_start = time.time()
    print(f"\n🔎 Searching Amazon for matching products...")
    amazon_results = amazon_search(product_details)
    search_time = time.time() - search_start
    print(f"✅ Amazon search completed in {search_time:.2f}s")
    timing['amazon_search'] = search_time
    
    # Check if we have results before proceeding
    if not amazon_results.get("results") or len(amazon_results["results"]) == 0:
        print("\n❌ No products found in Amazon search")
        return {
            "match_found": False,
            "top_match": None,
            "discrepancies": [],
            "timing": timing,
            "total_time": time.time() - start_total
        }
    
    # Verify product match with LLM using JSON mode
    verify_start = time.time()
    print(f"\n⚖️ Verifying matches...")
    verification_messages = [
        {"role": "system", "content": "Analyze the Amazon product results and determine how well they match the original product details. Assign a confidence score (0-1) to each result based on how closely it matches."},
        {"role": "user", "content": f"Compare these product details:\n\nOriginal product: {json.dumps(product_details)}\n\nAmazon results: {json.dumps(amazon_results)}\n\nFor each result, provide a confidence score and list any discrepancies. DO NOT include links or URLs in your response. Format your response as a JSON object with 'ranked_matches' (array of objects with 'asin', 'confidence_score', and 'discrepancies' fields) and 'general_discrepancies' (array of strings)."}
    ]
    
    verification_response = client.chat.completions.create(
        model="accounts/fireworks/models/mixtral-8x7b-instruct",
        messages=verification_messages,
        response_format={"type": "json_object"},
        max_tokens=1000
    )
    
    # Parse verification results with error handling
    try:
        llm_verification = json.loads(verification_response.choices[0].message.content)
        print(f"✅ Verification completed successfully")
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing verification response: {str(e)}")
        print(f"Raw response: {verification_response.choices[0].message.content[:500]}... (truncated)")
        # Create a default fallback structure
        llm_verification = {
            "ranked_matches": [
                {"asin": result.get("asin"), "confidence_score": 0.5, "discrepancies": []}
                for result in amazon_results.get("results", [])
            ],
            "general_discrepancies": []
        }
    
    verification_time = time.time() - verify_start
    print(f"⏱️ Verification took {verification_time:.2f}s")
    timing['verification'] = verification_time
    
    # Process and enhance the results with direct information from the Amazon API
    processing_start = time.time()
    print(f"\n🏁 Processing final results...")
    
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
    
    # Print final match result
    if final_results["top_match"]:
        print("\n✨ Found a match on Amazon!")
        print(f"   Title: {final_results['top_match']['title']}")
        if 'brand' in final_results['top_match']:
            print(f"   Brand: {final_results['top_match']['brand']}")
        
        # Handle price display correctly
        price = final_results['top_match']['price']
        if isinstance(price, dict) and 'raw' in price:
            print(f"   Price: {price['raw']}")
        elif price:
            print(f"   Price: {price}")
            
        # Print link only if it exists
        if final_results['top_match']['link']:
            print(f"   Link: {final_results['top_match']['link']}")
        else:
            print("   Link: Not available")
    else:
        print("\n❌ No suitable match found on Amazon")
    
    # Print overall timing
    print(f"\n⏱️ Total processing time: {timing['total_time']:.2f} seconds")
    

    # Clean up the temporary file
    os.unlink(image_path)
    
    return final_results

# Streamlit UI
st.title("Product Image Analyzer")
st.write("Upload a product image to search for it on Amazon")

# API key input (preferably use st.secrets in production)
api_key = st.text_input("Enter your Fireworks API key:", type="password")

# File uploader
uploaded_file = st.file_uploader("Choose a product image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None and api_key:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Process button
    if st.button("Analyze Product"):
        with st.spinner("Processing..."):
            results = process_product_image(uploaded_file, api_key)
        
        # Display results in a nice format
        if results["match_found"]:
            st.success("✅ Product match found!")
            
            # Create columns for product details
            col1, col2 = st.columns([1, 2])
            
            # Show product image in first column
            if 'top_match' in results and 'image' in results['top_match']:
                image_url = results['top_match']['image']
                if isinstance(image_url, dict) and 'link' in image_url:
                    image_url = image_url['link']
                col1.image(image_url, width=200)
            
            # Show details in second column
            if 'top_match' in results:
                match = results['top_match']
                col2.subheader(match.get('title', 'Product'))
                
                if 'brand' in match:
                    col2.write(f"**Brand:** {match['brand']}")
                
                # Handle price display
                price = match.get('price', '')
                if isinstance(price, dict) and 'raw' in price:
                    col2.write(f"**Price:** {price['raw']}")
                elif price:
                    col2.write(f"**Price:** {price}")
                
                # Product link
                if 'link' in match and match['link']:
                    col2.markdown(f"[View on Amazon]({match['link']})")
            
            # Discrepancies
            if results.get('discrepancies'):
                st.subheader("Potential discrepancies:")
                for disc in results['discrepancies']:
                    st.write(f"- {disc}")
            
        else:
            st.error("❌ No product match found")
            
        # Timing information (optional - can be hidden in a collapsible section)
        with st.expander("Processing details"):
            st.write(f"Total processing time: {results['timing']['total_time']:.2f} seconds")
            st.json(results)
else:
    st.info("Please upload an image and enter your API key to begin analysis")

