from fastapi import FastAPI, WebSocket
import uvicorn

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Clean WebSocket Service on 8309"}

@app.websocket("/ws/clean")
async def clean_websocket(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text("SUCCESS: Clean WebSocket on 8309!")
    
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Clean Echo: {data}")
    except:
        pass

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8309)
