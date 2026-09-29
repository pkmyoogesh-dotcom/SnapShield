"""Start the SnapShield server: python run.py"""
import uvicorn

HOST, PORT = "127.0.0.1", 8000
if __name__ == "__main__":
    print(f"Dashboard : http://{HOST}:{PORT}\nAPI       : http://{HOST}:{PORT}/api/v1\nSwagger   : http://{HOST}:{PORT}/docs")
    uvicorn.run("backend.main:app", host=HOST, port=PORT)
