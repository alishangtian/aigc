from uvicorn import run
from app.controller.vl import app

def start():
    run(app, host='0.0.0.0', port=8000)

if __name__ == '__main__':
    start()