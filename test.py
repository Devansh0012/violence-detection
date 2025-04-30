import asyncio
import websockets
import json

async def test_websocket():
    uri = "ws://localhost:8000/ws"
    print(f"Connecting to {uri}")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to WebSocket")
            
            # Send a test config message
            config = {"use_rtsp": False}
            print(f"Sending config: {config}")
            await websocket.send(json.dumps(config))
            
            # Wait for a response
            print("Waiting for response...")
            response = await websocket.recv()
            print(f"Received: {response}")
            
            # Keep the connection open a bit
            await asyncio.sleep(5)
            print("Test completed successfully")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket())