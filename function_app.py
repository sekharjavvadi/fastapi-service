import azure.functions as func
import logging
from fast_api import analyze_video_endpoint,object_detection_endpoint,tab_shift_endpoint,detect_multiple_voices_endpoint


app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="http_trigger")
def http_trigger(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    video_url = req.params.get('video_url')
    input_seconds = req.params.get('input_seconds')
    if not video_url and input_seconds:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            video_url = req_body.get('video_url')
            input_seconds = req_body.get('input_seconds')

    if video_url and input_seconds:
        result=analyze_video_endpoint(video_url,input_seconds)
        # return func.HttpResponse(result)
        return func.HttpResponse(f"{result}")
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a video_url in the query string or in the request body for a personalized response.",
             status_code=200
        )
    

@app.route(route="http_trigger_object_detection")
def http_trigger(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    video_url = req.params.get('video_url')
    input_seconds = req.params.get('input_seconds')
    if not video_url and input_seconds:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            video_url = req_body.get('video_url')
            input_seconds = req_body.get('input_seconds')

    if video_url and input_seconds:
        result=object_detection_endpoint(video_url,input_seconds)
        return func.HttpResponse(f"{result}")
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a video_url in the query string or in the request body for a personalized response.",
             status_code=200
        )


@app.route(route="http_trigger_tab_shift")
def http_trigger(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    video_url = req.params.get('video_url')
    input_seconds = req.params.get('input_seconds')
    if not video_url and input_seconds:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            video_url = req_body.get('video_url')
            input_seconds = req_body.get('input_seconds')

    if video_url and input_seconds:
        result=tab_shift_endpoint(video_url,input_seconds)
        return func.HttpResponse(f"{result}")
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a video_url in the query string or in the request body for a personalized response.",
             status_code=200
        )
#small comment

@app.route(route="http_trigger_multiple_voice")
def http_trigger(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    video_url = req.params.get('video_url')
    input_seconds = req.params.get('input_seconds')
    if not video_url and input_seconds:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            video_url = req_body.get('video_url')
            input_seconds = req_body.get('input_seconds')

    if video_url and input_seconds:
        result=detect_multiple_voices_endpoint(video_url,input_seconds)
        return func.HttpResponse(f"{result}")
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a video_url in the query string or in the request body for a personalized response.",
             status_code=200
        )