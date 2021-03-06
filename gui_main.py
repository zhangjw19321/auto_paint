import threading

import cv2
import tkinter as tk
from tkinter import *
from PIL import Image
from PIL import ImageTk
import sys
import os
current_path = os.getcwd()
env_path = os.path.join(current_path,"AttentionedDeepPaint")
print("add path:",env_path)
sys.path.append(env_path)
from colorize import *


# from ball import Ball
# from localization import Localization


class MainGUI:
    def __init__(self):
        self.font = ('Times', '24', 'bold')
        self.top = None
        self.basic = None
        self.pro = None
        self.face = None
        self.label_img = None
        self.face = None
        self.gesture = None
        self.body = None
        self.ball = None
        self.localizationTop = None
        self.auto_light = None
        self.ball_detecting = None
        self.screen_width = 980
        self.screen_height = 580
        self.size = "%dx%d+%d+%d" % (self.screen_width, self.screen_height, 0, 0)
        self.faceStopped = False
        self.gestureStopped = False
        self.bodyStopped = False
        self.ballStopped = False
        self.img_tk = None
        self.localizationProgram = None
        self.guide_map = None
        self.cap = cv2.VideoCapture("laohu2.mp4")
        self.take_capture = None
        self.camera_flag = False
        self.save_picture_flag = False
        self.save_frame = None
        self.painted_frame = None
        self.receiver_email = None
        self.painted_flag  = False
        self.makeTopGui()

    ################# level 1 GUI ##############################
    def makeTopGui(self):
        print("top gui starts")
        self.top = tk.Tk()
        self.screen_width, self.screen_height = self.getScreenSize(self.top)
        self.size = "%dx%d+%d+%d" % (self.screen_width, self.screen_height, 0, 0)
        self.top.geometry(self.size)
        button_basic = tk.Button(self.top, text="开始体验", font=self.font, command=self.callbackBasic)
        button_pro = tk.Button(self.top, text="PRO", font=self.font, command=self.callBackPro)
        button_basic.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        button_pro.pack(fill=tk.BOTH, expand=True, side=tk.RIGHT)
        welcome_pic = PhotoImage(file = 'data/renwu.png')
        welcome_label = Label(self.top, image = welcome_pic)
        # label2.bm = bm
        welcome_label.pack()
        welcome_label.place(x=int(self.screen_width/2),y=0)
        self.top.bind("<Escape>", self.end_top_fullscreen)
        self.top.bind("<F1>", self.start_top_fullscreen)
        self.top.mainloop()
        print("top gui ends")

    def callbackBasic(self):
        self.top.quit()
        self.top.destroy()
        self.makeBasicMenu()

    def callBackPro(self):
        pass
        # self.top.quit()
        # self.top.destroy()
        # self.makeProMenu()

    ###############################################################

    ############## level 2 GUI-- basic ####################################
    def makeBasicMenu(self):
        print("basic start")
        self.basic = tk.Tk()
        self.basic.geometry(self.size)
        frame_button1 = tk.Frame(self.basic)
        frame_button2 = tk.Frame(self.basic)
        button_face = tk.Button(frame_button1, text="开始拍照", font=self.font, command=self.call_back_take_picture)
        button_gesture = tk.Button(frame_button1, text="开始上色", font=self.font, command=self.call_back_auto_paint)
        button_body = tk.Button(frame_button2, text="动画生成", font=self.font, command=self.callbackBody)
        button_ball = tk.Button(frame_button2, text="动画下载", font=self.font, command=self.call_back_send_video)
        button_back = tk.Button(self.basic, text="返回上级", font=self.font, command=self.callbackBasicBack)
        frame_button1.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        frame_button2.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        button_face.pack(fill=tk.BOTH, expand=True, side=tk.TOP)
        button_gesture.pack(fill=tk.BOTH, expand=True, side=tk.BOTTOM)
        button_body.pack(fill=tk.BOTH, expand=True, side=tk.TOP)
        button_ball.pack(fill=tk.BOTH, expand=True, side=tk.BOTTOM)
        button_back.pack(fill=tk.BOTH, expand=True, side=tk.RIGHT)
        self.basic.mainloop()
        print("basic end")

    def call_back_take_picture(self):
        self.basic.quit()
        self.basic.destroy()
        self.make_take_picture_gui()

    def call_back_send_video(self):
        self.basic.quit()
        self.basic.destroy()
        self.make_send_email_gui()

    def call_back_auto_paint(self):
        self.basic.quit()
        self.basic.destroy()
        self.make_auto_paint_gui()  

    def callbackFace(self):
        self.basic.quit()
        self.basic.destroy()
        self.makeFaceGui()

    def callbackGesture(self):
        self.basic.quit()
        self.basic.destroy()
        self.makeGestureGui()

    def callbackBody(self):
        self.basic.quit()
        self.basic.destroy()
        self.makeBodyGui()

    def callbackBall(self):
        self.basic.quit()
        self.basic.destroy()
        self.makeBallGui()

    def callbackBasicBack(self):
        self.basic.quit()
        self.basic.destroy()
        self.makeTopGui()

    ##########################################################################

    #############  level 2 GUI -- pro #############################################
    def makeProMenu(self):
        self.pro = tk.Tk()
        self.pro.geometry(self.size)
        frame_button = tk.Frame(self.pro, width=int(self.screen_width / 3 * 2))
        button_localization = tk.Button(frame_button, text="Localization", font=self.font,
                                        command=self.callbackLocalization)
        button_football = tk.Button(frame_button, text="Football", font=self.font, command=self.callbackFootball)
        button_back = tk.Button(self.pro, text="BACK", font=self.font, command=self.callbackProBack)
        button_localization.pack(fill=tk.BOTH, expand=True, side=tk.TOP)
        button_football.pack(fill=tk.BOTH, expand=True, side=tk.BOTTOM)
        frame_button.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        button_back.pack(fill=tk.BOTH, expand=True, side=tk.RIGHT)
        self.pro.mainloop()

    def callbackLocalization(self):
        self.pro.quit()
        self.pro.destroy()
        self.localizationProgram = Localization()
        self.localizationProgram.startCamera()
        self.makeLocalizationGui()

    def callbackFootball(self):
        self.pro.quit()
        self.pro.destroy()
        self.makeFootballGui()

    def callbackProBack(self):
        self.pro.quit()
        self.pro.destroy()
        self.makeTopGui()

    ##########################################################################


    ############ level 3 GUI take pictures #############
    def make_take_picture_gui(self):
        self.face = tk.Tk()
        self.face.geometry(self.size)
        frame_button1 = tk.Frame(self.face)
        frame_button2 = tk.Frame(self.face)
        self.label_img = tk.Button(frame_button1, bg='black', state='disabled')
        self.label_img.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        button_back = tk.Button(frame_button2, text="返回上级", font=self.font, command=self.call_back_take_picture_back)
        button_save_picture = tk.Button(frame_button2, text="保存图片", font=self.font, command=self.save_and_show_picture)
        button_restart_capture = tk.Button(frame_button2, text="重新拍照", font=self.font, command=self.restart_take_picture)
        
        
        frame_button1.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        frame_button2.pack(fill=tk.BOTH, expand=True, side=tk.RIGHT)
        button_back.pack(fill=tk.BOTH, expand=True, side=tk.BOTTOM)
        button_save_picture.pack(fill=tk.BOTH, expand=True, side=tk.TOP)
        button_restart_capture.pack(fill=tk.BOTH, expand=True, side=tk.TOP)
        self.camera_flag = True
        self.take_capture = cv2.VideoCapture(0)
        self.take_picture_loop()
        self.face.mainloop()

    def save_and_show_picture(self):
        self.save_picture_flag = True
    def restart_take_picture(self):
        self.save_picture_flag = False

    def call_back_take_picture_back(self):
        self.face.quit()
        self.face.destroy()
        self.camera_flag = False
        self.take_capture.release()
        self.makeBasicMenu()

    def take_picture_loop(self):
        # print("come take picture",self.camera_flag,self.save_picture_flag)
        if self.camera_flag and not self.save_picture_flag:
            # cv_img = cv2.imread("face.jpg")
            ret,self.save_frame = self.take_capture.read()
            pil_image = Image.fromarray(cv2.cvtColor(self.save_frame, cv2.COLOR_BGR2RGB))
            pil_image = pil_image.resize((int(self.screen_width / 2), int(self.screen_height / 2)))
            self.img_tk = ImageTk.PhotoImage(pil_image)
            self.label_img.config(image=self.img_tk)
            self.face.after(100, self.take_picture_loop)
        elif self.camera_flag and self.save_picture_flag:
            pil_image = Image.fromarray(cv2.cvtColor(self.save_frame, cv2.COLOR_BGR2RGB))
            pil_image = pil_image.resize((int(self.screen_width / 2), int(self.screen_height / 2)))
            self.img_tk = ImageTk.PhotoImage(pil_image)
            self.label_img.config(image=self.img_tk)
            self.face.after(100, self.take_picture_loop)

    ############# level 3 GUI -- auto paint ##############################
    def make_auto_paint_gui(self):
        self.body = tk.Tk()
        self.body.geometry(self.size)
        self.label_img = tk.Label(self.body, bg='black', width=int(self.screen_width / 16), state='disabled')
        self.label_img.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        button_back = tk.Button(self.body, text="返回上级", font=self.font, command=self.callbackBodyBack)
        button_back.pack(fill=tk.BOTH, expand=True, side=tk.RIGHT)
        self.bodyStopped = False
        self.auto_paint_loop()
        self.body.mainloop()
    def callbackBodyBack(self):
        self.body.quit()
        self.body.destroy()
        self.bodyStopped = True
        self.makeBasicMenu()
    
    def auto_paint_image(self):
        if not self.painted_flag:
            self.painted_frame = paint_color(self.save_frame)

            self.painted_flag = True
    def auto_paint_loop(self):
        if self.save_frame is not None:
        # start to paint
            self.auto_paint_image()
            cv_img = self.painted_frame
            print("save image",self.save_frame)
            print("painted frame ",self.painted_frame)
            if cv_img is not None:
                pil_image = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
                pil_image = pil_image.resize((int(self.screen_width / 2), int(self.screen_height / 2)))
                self.img_tk = ImageTk.PhotoImage(pil_image)
                self.label_img.config(image=self.img_tk)
                self.body.after(100, self.auto_paint_loop)
    def auto_paint_loop(self):
        
        cv_img = cv2.imread("./data/renwu.png")
        print("image  ",cv_img)
        pil_image = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        pil_image = pil_image.resize((int(self.screen_width / 2), int(self.screen_height / 2)))
        self.img_tk = ImageTk.PhotoImage(pil_image)
        self.label_img.config(image=self.img_tk)
        self.body.after(100, self.auto_paint_loop)

    ############# level 3 GUI- face #######################################
    def makeFaceGui(self):
        self.face = tk.Tk()
        self.face.geometry(self.size)
        self.label_img = tk.Button(self.face, bg='black', width=int(self.screen_width / 16), state='disabled')
        self.label_img.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        button_back = tk.Button(self.face, text="BACK", font=self.font, command=self.callbackFaceBack)
        button_back.pack(fill=tk.BOTH, expand=True, side=tk.RIGHT)
        self.camera_flag = True
        self.faceLoop()
        self.face.mainloop()

    def call_back_face_back(self):
        self.face.quit()
        self.face.destroy()
        self.camera_flag = False
        self.makeBasicMenu()

    def Faceloop(self):
        print("come face loop")
        if not self.faceStopped:
            # cv_img = cv2.imread("face.jpg")
            ret,cv_img = self.cap.read()
            if not ret:
                pass
            else:
                pil_image = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
                pil_image = pil_image.resize((int(self.screen_width / 2), int(self.screen_height / 2)))
                self.img_tk = ImageTk.PhotoImage(pil_image)
                self.label_img.config(image=self.img_tk)
                self.face.after(100, self.faceLoop)
    ############# level 3 GUI -- send video via email #################

    def make_send_email_gui(self):
        self.gesture = tk.Tk()
        self.gesture.geometry(self.size)
        frame_button1 = tk.Frame(self.gesture)
        frame_button2 = tk.Frame(self.gesture)
        self.label_img = tk.Button(frame_button1, bg='black', width=int(self.screen_width /2), state='disabled')
        self.label_img.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        self.my_entry_box = tk.Entry(frame_button2)
        button_back = tk.Button(frame_button2, text="返回上级", font=self.font, command=self.callbackGestureBack)
        button_send = tk.Button(frame_button2, text="动画发送", font=self.font, command=self.send_email)
        frame_button1.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        frame_button2.pack(fill=tk.BOTH, expand=True, side=tk.RIGHT)
        self.my_entry_box.pack(fill=tk.BOTH, expand=True, side=tk.TOP)
        button_back.pack(fill=tk.BOTH, expand=True, side=tk.BOTTOM)
        button_send.pack(fill=tk.BOTH, expand=True, side=tk.BOTTOM)
        self.gestureStopped = False
        self.show_painted_frame()
        self.gesture.mainloop()
    def call_back_send_email_back(self):
        self.gesture.quit()
        self.gesture.destroy()
        self.gestureStopped = True
        self.makeBasicMenu()

    def show_painted_frame(self):
        if not self.gestureStopped:
            cv_img = self.save_frame
            pil_image = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
            pil_image = pil_image.resize((int(self.screen_width / 2), int(self.screen_height / 2)))
            self.img_tk = ImageTk.PhotoImage(pil_image)
            self.label_img.config(image=self.img_tk)
            self.gesture.after(100, self.gestureLoop)

    def send_email(self):
        import email, smtplib, ssl
        from email import encoders
        from email.mime.base import MIMEBase
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        self.receiver_email = self.my_entry_box.get()
        print("receiver email is: ", self.receiver_email)
        subject = "Animate Video"
        body = "This is an email with attachment"
        sender_email = "z_jun_wei@163.com"
        receiver_email = self.receiver_email
        password = "2012weili01"
        # Create a multipart message and set headers
        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = receiver_email
        message["Subject"] = subject
        message["Bcc"] = receiver_email  # Recommended for mass emails

        # Add body to email
        message.attach(MIMEText(body, "plain"))

        filename = "data/laohu2.mp4"  # In same directory as script

        # Open PDF file in binary mode
        with open(filename, "rb") as attachment:
            # Add file as application/octet-stream
            # Email client can usually download this automatically as attachment
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
        # Encode file in ASCII characters to send by email    
        encoders.encode_base64(part)

        # Add header as key/value pair to attachment part
        part.add_header(
            "Content-Disposition",
            f"attachment; filename= {filename}",
        )
        # Add attachment to message and convert message to string
        message.attach(part)
        text = message.as_string()
        # Log in to server using secure context and send email
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.163.com",994) as server:
            server.login(sender_email, password)
            # server.sendmail(sender_email, receiver_email, text)
            print("send finish")


    ############# level 3 GUI- gesture #######################################
    def makeGestureGui(self):
        self.gesture = tk.Tk()
        self.gesture.geometry(self.size)
        self.label_img = tk.Button(self.gesture, bg='black', width=int(self.screen_width / 16), state='disabled')
        self.label_img.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        button_back = tk.Button(self.gesture, text="BACK", font=self.font, command=self.callbackGestureBack)
        button_back.pack(fill=tk.BOTH, expand=True, side=tk.RIGHT)
        self.gestureStopped = False
        # self.gestureLoop()
        self.gesture.mainloop()

    def callbackGestureBack(self):
        self.gesture.quit()
        self.gesture.destroy()
        self.gestureStopped = True
        self.makeBasicMenu()

    def gestureLoop(self):
        if not self.gestureStopped:
            cv_img = cv2.imread("newmap.jpg")
            pil_image = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
            pil_image = pil_image.resize((int(self.screen_width / 2), int(self.screen_height / 2)))
            self.img_tk = ImageTk.PhotoImage(pil_image)
            self.label_img.config(image=self.img_tk)
            self.gesture.after(100, self.gestureLoop)

   
    ############# level 3 GUI- ball #######################################
    def makeBallGui(self):
        self.ball = tk.Tk()
        self.ball.geometry(self.size)
        self.label_img = tk.Label(self.ball, bg='black', width=int(self.screen_width / 16))
        self.label_img.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        button_back = tk.Button(self.ball, text="BACK", font=self.font, command=self.callbackBallBack)
        button_back.pack(fill=tk.BOTH, expand=True, side=tk.RIGHT)
        self.ballStopped = False
        self.ball_detecting = Ball(self.screen_width, self.screen_height)
        t = threading.Thread(target=self.ball_detecting.detect)
        t.setDaemon(True)
        t.start()
        # self.ball.after(1000, self.ballLoop)
        self.ball.mainloop()

    def callbackBallBack(self):
        self.ball.quit()
        self.ball.destroy()
        self.ballStopped = True
        self.ball_detecting.ballStopped = True
        self.makeBasicMenu()

    def ballLoop(self):
        print("in ball loop")
        if not self.ballStopped:
            cv_img = self.ball_detecting.ball_frame
            if cv_img is not None:
                cv2.imshow("ball", cv_img)
            cv2.waitKey(1)

            pil_image = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
            pil_image = pil_image.resize((int(self.screen_width / 2), int(self.screen_height / 2)))
            self.img_tk = ImageTk.PhotoImage(pil_image)
            self.label_img.config(image=self.img_tk)
            self.ball.after(100, self.ballLoop)

    ##########################################################################################################

    ############### level 3 GUI - pro - localization ########################################################
    def makeLocalizationGui(self):
        self.localizationTop = tk.Tk()
        self.localizationTop.geometry(self.size)
        img_open = Image.open('LOCALIZATION/guide_map.jpg').resize((int(self.screen_width / 2), self.screen_height),
                                                                   Image.ANTIALIAS)
        img_tk = ImageTk.PhotoImage(img_open)
        label_logo = tk.Label(self.localizationTop, image=img_tk)
        label_logo.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        frame_buttons = tk.Frame(self.localizationTop)
        button_autolight = tk.Button(frame_buttons, text="AUTO LIGHT", font=self.font, command=self.callBackAutoLight)
        button_start = tk.Button(frame_buttons, text="START", font=self.font, command=self.callBackLocalStart)
        button_back = tk.Button(frame_buttons, text="BACK", font=self.font, command=self.callBackLocalBack)
        button_autolight.pack(fill=tk.BOTH, expand=True, side=tk.TOP)
        button_start.pack(fill=tk.BOTH, expand=True)
        button_back.pack(fill=tk.BOTH, expand=True, side=tk.BOTTOM)
        frame_buttons.pack(fill=tk.BOTH, expand=True, side=tk.RIGHT)

        self.localizationTop.resizable(0, 0)

        self.localizationTop.mainloop()

    def callBackAutoLight(self):
        self.localizationProgram.LEGO_Video_shower.autoColorStopped = False
        self.localizationProgram.LEGO_Video_shower.autoColorCancled = False
        t = threading.Thread(target=self.localizationProgram.autoLight)
        t.setDaemon(True)
        t.start()
        self.localizationTop.quit()
        self.localizationTop.destroy()
        self.makeAutoLightGui()

    def callBackLocalStart(self):
        self.localizationTop.quit()
        self.localizationTop.destroy()
        self.makeGuideMap()
        pass

    def callBackLocalBack(self):
        self.localizationTop.quit()
        self.localizationTop.destroy()
        self.localizationProgram.stopCamera()
        self.makeProMenu()

    #########################################################################################################

    ############## level 4 GUI - pro- localization - auto light ###############################################
    def makeAutoLightGui(self):
        self.auto_light = tk.Tk()
        self.auto_light.geometry(self.size)
        self.label_img = tk.Label(self.auto_light)
        self.label_img.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        frame_buttons = tk.Frame(self.auto_light, width=int(self.screen_width / 3 * 2), height=int(self.screen_height))
        auto_light_confirm = tk.Button(frame_buttons, text="CONFIRM", font=self.font,
                                       command=self.CallBackAutoLightConfirm)
        auto_light_cancel = tk.Button(frame_buttons, text="CANCEL", font=self.font,
                                      command=self.CallBackAutoLightCancel)
        auto_light_confirm.pack(fill=tk.BOTH, expand=True)
        auto_light_cancel.pack(fill=tk.BOTH, expand=True)
        frame_buttons.pack(fill=tk.BOTH, expand=True, side=tk.RIGHT)
        self.auto_light.after(200, self.auto_light_video_loop)
        self.auto_light.mainloop()

    def auto_light_video_loop(self):
        if not self.localizationProgram.LEGO_Video_shower.autoColorStopped:
            cv_img = self.localizationProgram.LEGO_Video_shower.AutoColorImage
            pil_image = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
            pil_image = pil_image.resize((int(self.screen_width / 2), int(self.screen_height / 2)))
            self.img_tk = ImageTk.PhotoImage(pil_image)
            self.label_img.config(image=self.img_tk)
            self.auto_light.after(100, self.auto_light_video_loop)

    def CallBackAutoLightConfirm(self):
        self.localizationProgram.LEGO_Video_shower.autoColorStopped = True
        self.auto_light.quit()
        self.auto_light.destroy()
        self.makeLocalizationGui()

    def CallBackAutoLightCancel(self):
        self.localizationProgram.LEGO_Video_shower.autoColorCancled = True
        self.localizationProgram.LEGO_Video_shower.autoColorStopped = True
        self.auto_light.quit()
        self.auto_light.destroy()
        self.makeLocalizationGui()

    #############level 4 GUI - pro - localization - guide map ##############################################
    def makeGuideMap(self):
        self.guide_map = tk.Tk()
        self.guide_map.geometry(self.size)
        self.label_img = tk.Label(self.guide_map)
        frame_buttons = tk.Frame(self.guide_map)
        self.button_display_inst = tk.Label(frame_buttons, text="READ TO GO", font=self.font)
        button_map_quit = tk.Button(frame_buttons, text="QUIT", font=self.font, command=self.mapQuitCallBack)
        self.button_display_inst.pack(fill=tk.BOTH, expand=True, side=tk.TOP)
        button_map_quit.pack(fill=tk.BOTH, expand=True)
        self.label_img.pack(expand=True, side=tk.LEFT)
        frame_buttons.pack(fill=tk.BOTH, expand=True, side=tk.RIGHT)
        self.label_img.bind("<Button-1>", self.map_guide_order)
        t = threading.Thread(target=self.localizationProgram.startWork)
        t.setDaemon(True)
        t.start()
        self.map_video_loop()
        self.guide_map.mainloop()

    def mapQuitCallBack(self):
        self.localizationProgram.quit_flag = True
        self.localizationProgram.LEGO_Video_shower.guideMapStopped = True
        self.guide_map.quit()
        self.guide_map.destroy()
        self.makeLocalizationGui()

    def map_video_loop(self):
        if not self.localizationProgram.LEGO_Video_shower.guideMapStopped:
            cv_img = self.localizationProgram.LEGO_Video_shower.visualImage
            pil_image = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
            pil_image = pil_image.resize((int(self.screen_width / 2), int(self.screen_width / 2)))
            self.img_tk = ImageTk.PhotoImage(pil_image)
            self.label_img.config(image=self.img_tk)
            self.guide_map.after(100, self.map_video_loop)

    def map_guide_order(self, event):
        w, h = self.getWindowSize(self.label_img)
        x = event.x
        y = event.y
        if y < h / 3:
            if x < w / 3:
                self.localizationProgram.target_index = 1
            elif x < w / 3 * 2:
                self.localizationProgram.target_index = 2
            else:
                self.localizationProgram.target_index = 3
        elif y > h / 3 * 2:
            if x < w / 3:
                self.localizationProgram.target_index = 4
            elif x < w / 3 * 2:
                self.localizationProgram.target_index = 5
            else:
                self.localizationProgram.target_index = 6
        self.button_display_inst.config(
            text="TARGET=" + str(self.localizationProgram.target_index))  # +'\n'+str(x)+","+str(y)+"\n"+str(w)+","+str(h))

    ############## level 3 GUI - pro - football ##############################################################
    def makeFootballGui(self):
        self.football = tk.Tk()
        self.football.geometry(self.size)
        self.label_img = tk.Label(self.football, bg='black', width=int(self.screen_width / 16))
        self.label_img.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        frame_button = tk.Frame(self.football)
        self.button_start = tk.Button(frame_button, text='START', font=self.font, command=self.callBackFootballStart)
        self.button_start.pack(fill=tk.BOTH, expand=True, side=tk.TOP)
        button_back = tk.Button(frame_button, text="BACK", font=self.font, command=self.callbackFootballBack)
        button_back.pack(fill=tk.BOTH, expand=True, side=tk.BOTTOM)
        frame_button.pack(fill=tk.BOTH, expand=True, side=tk.RIGHT)
        self.ballStopped = False
        self.ball_detecting = Ball(self.screen_width, self.screen_height)
        t = threading.Thread(target=self.ball_detecting.detect)
        t.setDaemon(True)
        t.start()

        self.football.mainloop()

        pass

    def callBackFootballStart(self):
        t = threading.Thread(target=self.ball_detecting.ball_tracking_robot)
        t.setDaemon(True)
        t.start()
        self.button_start.config(state="disabled", text="RUNNING")

    def callbackFootballBack(self):
        self.football.quit()
        self.football.destroy()
        self.ballStopped = True
        self.ball_detecting.ballStopped = True
        self.makeProMenu()

    ########################################################################################################

    ##################   tool functions   ###############################
    def start_top_fullscreen(self, event=None):
        self.top.attributes("-fullscreen", True)
        return "break"

    def end_top_fullscreen(self, event=None):
        self.top.attributes("-fullscreen", False)
        return "break"

    def getScreenSize(self, window):
        return window.winfo_screenwidth(), window.winfo_screenheight()

    def getWindowSize(self, window):
        return window.winfo_reqwidth(), window.winfo_reqheight()


#######################################################################


if __name__ == '__main__':
    gui = MainGUI()
