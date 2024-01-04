from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import socketio


        
# Initialize FastAPI instance after route declaration
app = FastAPI()

# Move the route declaration here
@app.post("/process_name")
async def process_name(request: Request):
    data = await request.json()
    name = data.get("name", "")
    print(f"Received name from client: {name}")
    
    # Implement your logic for processing the name (e.g., store it, perform an action, etc.)
    
    return JSONResponse(content={"message": "Name processed successfully"})
