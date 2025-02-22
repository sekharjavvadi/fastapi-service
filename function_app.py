import azure.functions as func
import logging
from fast_api import analyze_video_endpoint,tab_shift_endpoint


app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="http_trigger_analysis", auth_level=func.AuthLevel.ANONYMOUS)
def http_trigger_analysis(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
   
    video_url = req.params.get('video_url')
    input_seconds = req.params.get('input_seconds')
 
    if not video_url or input_seconds:
        try:
            req_body = req.get_json()
            video_url = req_body.get('video_url')
            input_seconds = req_body.get('input_seconds')
        except ValueError:
            pass
 
    if video_url or input_seconds:
        result = analyze_video_endpoint(video_url)
        return func.HttpResponse(f"{result}")
    else:
        return func.HttpResponse(
            "This HTTP triggered function executed successfully. Pass a video_url in the query string or in the request body for a personalized response.",
            status_code=200
        )
    

@app.route(route="http_trigger_tab_shift", auth_level=func.AuthLevel.ANONYMOUS)
def http_trigger_tab_shift(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
   
    video_url = req.params.get('video_url')
    input_seconds = req.params.get('input_seconds')
 
    if not video_url or input_seconds:
        try:
            req_body = req.get_json()
            video_url = req_body.get('video_url')
            input_seconds = req_body.get('input_seconds')
        except ValueError:
            pass
 
    if video_url or input_seconds:
        result = tab_shift_endpoint(video_url)
        return func.HttpResponse(f"{result}")
    else:
        return func.HttpResponse(
            "This HTTP triggered function executed successfully. Pass a video_url in the query string or in the request body for a personalized response.",
            status_code=200
        )