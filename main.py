import flet as ft
import cv2
import base64

from threading import Timer,Thread
from detect import run
from fire import Fire

cap = None
is_save_data = None
acc_max = None

def main(page: ft.Page):
    page.title = "OpenCV 视频流与 Flet 示例"
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

    def to_base64(image):
        base64_image = cv2.imencode('.png', image)[1]
        base64_image = base64.b64encode(base64_image).decode('utf-8') 
        return base64_image

    frame_control = None

    def update_frame(frame):
        nonlocal frame_control

        if frame_control is None:
            frame_control = ft.Image(
            width=640,
            height=480,
            fit=ft.ImageFit.CONTAIN,
            src_base64=to_base64(frame)
        )
            col.controls.append(frame_control)
        else:
            frame_control.src_base64 = to_base64(frame)
        
        col.update()
    
    def show_notify():
        VALUE = "检测到跌倒"

        if notify.value == VALUE:
            return
        
        notify.value = VALUE
        notify.update()

        def cancel():
            notify.value = ""
            notify.update()

        Timer(1, cancel).start()
    
    Thread(target=run,args=(cap,update_frame,show_notify,is_save_data,acc_max,vmax), daemon=True).start()


    notify = ft.Text("")
    
    page.add(
        (col:=ft.Column(
            controls=[
                notify,
            ],
            alignment=ft.MainAxisAlignment.CENTER,
        ))
    )

def start(video_source:str|int,_is_save_data:bool = False,_acc_max:int=58,_vmax=-10):
    global cap,is_save_data,acc_max,vmax
    is_save_data = _is_save_data
    acc_max = _acc_max
    vmax = _vmax

    try:
        cap = cv2.VideoCapture(video_source)
        ft.app(target=main)
    finally:
        cap.release()

if __name__ == "__main__":
    Fire(start)

