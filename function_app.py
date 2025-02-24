import azure.functions as func
import logging
from fast_api import analyze_video_endpoint, object_detection_endpoint, tab_shift_endpoint, detect_multiple_voices_endpoint

app = func.FunctionApp()

@app.route(route="http_trigger_analysis", auth_level=func.AuthLevel.ANONYMOUS)
def http_trigger_analysis(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    video_url = req.params.get('video_url')
    input_seconds = req.params.get('input_seconds')

    if not video_url or not input_seconds:
        try:
            req_body = req.get_json()
            video_url = req_body.get('video_url')
            input_seconds = req_body.get('input_seconds')
        except ValueError:
            pass

    if video_url and input_seconds:
        try:
            input_seconds = int(input_seconds)  # Ensure input_seconds is an integer
        except ValueError:
            return func.HttpResponse("Invalid input_seconds. Must be an integer.", status_code=400)
        
        result = analyze_video_endpoint(video_url, input_seconds)
        return func.HttpResponse(result, mimetype="application/json")
    
    return func.HttpResponse(
        "Invalid request. Please provide both video_url and input_seconds.",
        status_code=400
    )

@app.route(route="http_trigger_object_detection", auth_level=func.AuthLevel.ANONYMOUS)
def http_trigger_object_detection(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Processing object detection request.')

    video_url = req.params.get('video_url')

    if not video_url:
        try:
            req_body = req.get_json()
            video_url = req_body.get('video_url')
        except ValueError:
            pass

    if video_url:
        result = object_detection_endpoint(video_url)
        return func.HttpResponse(result, mimetype="application/json")

    return func.HttpResponse(
        "Invalid request. Please provide a valid video_url.",
        status_code=400
    )

@app.route(route="http_trigger_tab_shift", auth_level=func.AuthLevel.ANONYMOUS)
def http_trigger_tab_shift(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Processing tab shift detection request.')

    video_url = req.params.get('video_url')
    input_seconds = req.params.get('input_seconds')

    if not video_url or not input_seconds:
        try:
            req_body = req.get_json()
            video_url = req_body.get('video_url')
            input_seconds = req_body.get('input_seconds')
        except ValueError:
            pass

    if video_url and input_seconds:
        try:
            input_seconds = int(input_seconds)
        except ValueError:
            return func.HttpResponse("Invalid input_seconds. Must be an integer.", status_code=400)

        result = tab_shift_endpoint(video_url, input_seconds)
        return func.HttpResponse(result, mimetype="application/json")

    return func.HttpResponse(
        "Invalid request. Please provide both video_url and input_seconds.",
        status_code=400
    )
import json
import logging
import azure.functions as func

@app.route(route="http_trigger_multiple_voices", auth_level=func.AuthLevel.ANONYMOUS)
def http_trigger_multiple_voices(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Processing request for multiple voice detection.')

    # Extract video_url from query parameters or request body
    video_url = req.params.get('video_url')

    if not video_url:
        try:
            req_body = req.get_json()
            video_url = req_body.get('video_url')
        except ValueError:
            logging.error("Failed to parse JSON request body.")
            return func.HttpResponse(
                json.dumps({"error": "Invalid JSON format."}),
                mimetype="application/json",
                status_code=400
            )

    if not video_url:
        logging.warning("No video_url provided in request.")
        return func.HttpResponse(
            json.dumps({"error": "Invalid request. Please provide a valid video_url."}),
            mimetype="application/json",
            status_code=400
        )

    try:
        # Call the detection function
        result = detect_multiple_voices_endpoint(video_url)
        
        # Ensure result is a JSON string
        return func.HttpResponse(
            json.dumps(result),  # Serialize the response
            mimetype="application/json"
        )
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": "Internal server error", "details": str(e)}),
            mimetype="application/json",
            status_code=500
        )
