from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()
@app.get("/")
def index():
    return {"name":"Shrey"}

@app.post("/process_name")
async def process_name(request: Request):
    data = await request.json()
    name = data.get("name", "")
    print(f"Received name from client: {name}")
    
    # Implement your logic for processing the name (e.g., store it, perform an action, etc.)
    
    return JSONResponse(content={"message": "Name processed successfully"})

@app.post("/landmarks")
async def landmarks(request: Request):
    data = await request.json()
    
    # Print received landmark data on the terminal
    print("Received landmark data:")
    print(f"Left Shoulder X: {data.get('leftShoulderX', 0.0)}")
    print(f"Left Shoulder Y: {data.get('leftShoulderY', 0.0)}")
    print(f"Left Shoulder Z: {data.get('leftShoulderZ', 0.0)}")

    # Extract landmark data from the request JSON
    left_shoulder_x = data.get("leftShoulderX", 0.0)
    left_shoulder_y = data.get("leftShoulderY", 0.0)
    left_shoulder_z = data.get("leftShoulderZ", 0.0)

    # Implement your logic for processing the landmark data
    # Example: Store the landmark data in a database

    return JSONResponse(content={"message": "Landmarks processed successfully"})

if __name__=="__main__":
    uvicorn.run(app)
