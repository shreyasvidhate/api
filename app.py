from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json
import os
import subprocess
import requests

from Model.data_prediction_updated import receive_landmarks

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

    try:
        headers = {'Content-Type': 'application/json'}
        response = requests.post('https://api-mlkit.onrender.com/landmarks',headers=headers, json=data)
        print("before status")
        response.raise_for_status()
        print("after status")
        receive_landmarks()
        return JSONResponse(content={"message": "Landmarks processed successfully"})
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to send data to prediction script: {str(e)}")


    # write_coordinates(data,os.path.join(os.getcwd(),'coordinates.json'))
    # pred_path = os.path.join(os.getcwd(),'Model','data_prediction_updated.py')
    # subprocess.run(["python", pred_path, json.dumps(data)])

    return JSONResponse(content={"message": "Landmarks processed successfully"})

@app.post("/get_output")
async def process_name(request: Request):
    data = await request.json()
    print("Received data:", data)
    
    return JSONResponse(content={"message": "Received response"})

def write_coordinates(coordinates, file_path):
    with open(file_path, 'w') as file:
        json.dump(coordinates, file)

if __name__=="__main__":
    uvicorn.run(app)
