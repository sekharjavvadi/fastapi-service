import azure.functions as func
import logging
from fast_api import analyze_video_endpoint

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="http_trigger", methods=["POST"])
def http_trigger(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    try:
        req_body = req.get_json()
        video_url = req_body.get('video_url', None)
    except ValueError:
        return func.HttpResponse(
            "Invalid request body. Please provide a valid JSON object.",
            status_code=400
        )

    if video_url:
        result = analyze_video_endpoint(video_url)
        return func.HttpResponse(f"{result}")
    else:
        return func.HttpResponse(
            "Please provide a 'video_url' in the request body.",
            status_code=400
        )
