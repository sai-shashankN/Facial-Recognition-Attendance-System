import os
import shutil
import random
import pandas as pd
import face_recognition
from datetime import datetime
import tkinter as tk
from tkinter import Label, Button, ttk, Entry, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import cv2
import numpy as np
import pickle
from shutil import SameFileError

# Paths
BASE_DIR        = 'C:/Users/LENOVO/OneDrive/Desktop/INT345 PROJ/archive (2)'
FACES_DIR       = os.path.join(BASE_DIR, 'Faces', 'Faces')
ORIGINAL_DIR    = os.path.join(BASE_DIR, 'Original Images', 'Original Images')
INSERTED_DIR    = os.path.join(BASE_DIR, 'Inserted Images')
CSV_FILE        = os.path.join(BASE_DIR, 'Dataset.csv')
ATTENDANCE_FILE = os.path.join(BASE_DIR, 'attendance.csv')
ENCODINGS_FILE  = os.path.join(BASE_DIR, 'face_encodings.pkl')
TEMP_CAPTURE    = os.path.join(BASE_DIR, 'temp_capture.jpg')

os.makedirs(INSERTED_DIR, exist_ok=True)

known_face_encodings = []
known_face_names     = []
df = pd.read_csv(CSV_FILE)

def mark_attendance(name):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    rec = pd.DataFrame([{'Name': name, 'Timestamp': ts}])
    if not os.path.exists(ATTENDANCE_FILE):
        rec.to_csv(ATTENDANCE_FILE, index=False)
    else:
        rec.to_csv(ATTENDANCE_FILE, mode='a', header=False, index=False)

def load_face_encodings(pv, pb, lbl, callback):
    total = len(df)
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, 'rb') as f:
            encs, names = pickle.load(f)
            known_face_encodings.extend(encs)
            known_face_names.extend(names)
        callback(); return

    def worker():
        for i, row in enumerate(df.itertuples(), 1):
            path = os.path.join(FACES_DIR, row.id)
            img = face_recognition.load_image_file(path)
            encs = face_recognition.face_encodings(img)
            if encs:
                known_face_encodings.append(encs[0])
                known_face_names.append(row.label)
            pct = i/total*100
            pv.set(pct)
            lbl.after(0, lambda p=pct: lbl.config(text=f"Loading: {p:.1f}%"))
            pb.after(0, pb.update_idletasks)
        with open(ENCODINGS_FILE, 'wb') as f:
            pickle.dump((known_face_encodings, known_face_names), f)
        callback()

    threading.Thread(target=worker, daemon=True).start()

def start_real_webcam():
    # 1) Capture one frame
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        messagebox.showerror("Webcam Error", "Could not capture from webcam.")
        return show_launcher_gui()

    # 2) Save temp (optional)
    cv2.imwrite(TEMP_CAPTURE, frame)

    # 3) Try recognition
    try:
        rgb = frame[:, :, ::-1]
        locs = face_recognition.face_locations(rgb)
        encs = face_recognition.face_encodings(rgb, locs)

        names = []
        for enc in encs:
            matches = face_recognition.compare_faces(known_face_encodings, enc)
            dists   = face_recognition.face_distance(known_face_encodings, enc)
            idx     = np.argmin(dists) if dists.size else None
            if idx is not None and matches[idx]:
                name = known_face_names[idx]
                mark_attendance(name)
            else:
                name = "Unknown"
            names.append(name)

        # Draw boxes & labels
        for (top, right, bottom, left), name in zip(locs, names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
            cv2.putText(frame, name, (left, top-10),
                        cv2.FONT_HERSHEY_SIMPLEX, .75, (255,255,255), 2)

        # Popup result
        result = ", ".join(set(names)) if names else "No faces detected"
        messagebox.showinfo("Recognition Result", result)

    except Exception:
        messagebox.showinfo("Recognition Result", "Unknown")

    # 4) Show the annotated image
    cv2.imshow("Captured Recognition", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 5) Back to launcher
    show_launcher_gui()

def show_launcher_gui():
    r = tk.Tk()
    r.title("Select Mode")
    r.geometry("300x200")
    Label(r, text="Choose Mode", font=("Arial",14)).pack(pady=20)
    Button(r, text="Simulated Webcam", command=lambda:[r.destroy(), create_simulated_gui()]).pack(pady=5)
    Button(r, text="Real Webcam (Capture Once)", command=lambda:[r.destroy(), start_real_webcam()]).pack(pady=5)
    Button(r, text="Exit", command=r.destroy).pack(pady=5)
    r.mainloop()

def create_simulated_gui():
    root = tk.Tk()
    root.title("Simulated Webcam - Smart Attendance")
    root.geometry("500x650")

    Label(root, text="Search", font=("Arial",12)).pack(pady=5)
    sv = tk.StringVar()
    ent = Entry(root, textvariable=sv); ent.pack(pady=5)
    def filt(ev=None):
        q = sv.get().lower()
        dropdown['values'] = [n for n in known_face_names if q in n.lower()]
    ent.bind("<KeyRelease>", filt)

    Label(root, text="Select a Person", font=("Arial",14)).pack(pady=5)
    nv = tk.StringVar()
    dropdown = ttk.Combobox(root, textvariable=nv, values=known_face_names, state='readonly')
    dropdown.pack(pady=5)

    thumbs = tk.Frame(root); thumbs.pack(pady=10, fill='x')
    disp_img = Label(root); disp_img.pack(pady=10)
    disp_info = Label(root, text="", font=("Arial",12)); disp_info.pack(pady=5)

    def show_details(path, person):
        img = Image.open(path).resize((200,200))
        ph = ImageTk.PhotoImage(img)
        disp_img.config(image=ph); disp_img.image = ph
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        disp_info.config(text=f"Name: {person}\nTime: {ts}")
        mark_attendance(person)

    def on_name(ev=None):
        for w in thumbs.winfo_children(): w.destroy()
        person = nv.get()
        fld = os.path.join(ORIGINAL_DIR, person)
        if not os.path.isdir(fld): return
        files = [f for f in os.listdir(fld) if f.lower().endswith(('.jpg','.png','.jpeg'))]
        for fn in files:
            p = os.path.join(fld, fn)
            img = Image.open(p).resize((60,60))
            ph = ImageTk.PhotoImage(img)
            btn = Button(thumbs, image=ph, command=lambda p=p, per=person: show_details(p,per))
            btn.image = ph; btn.pack(side='left',padx=5,pady=5)

    dropdown.bind("<<ComboboxSelected>>", on_name)

    def insert_image():
        f = filedialog.askopenfilename(filetypes=[("Images","*.jpg *.png *.jpeg")])
        if not f: return
        dest = os.path.join(INSERTED_DIR, os.path.basename(f))
        try: shutil.copy(f, dest)
        except SameFileError: pass
        inserted_dd['values'] = os.listdir(INSERTED_DIR)

    Button(root, text="Insert Image", command=insert_image).pack(pady=5)
    Label(root, text="Inserted Images", font=("Arial",14)).pack(pady=5)
    iv = tk.StringVar()
    inserted_dd = ttk.Combobox(root, textvariable=iv, values=os.listdir(INSERTED_DIR), state='readonly')
    inserted_dd.pack(pady=5)

    def on_inserted(ev=None):
        fn = iv.get()
        path = os.path.join(INSERTED_DIR, fn)
        img = face_recognition.load_image_file(path)
        encs = face_recognition.face_encodings(img)
        if not encs:
            disp_info.config(text="No face found."); return
        enc = encs[0]
        matches = face_recognition.compare_faces(known_face_encodings, enc)
        dists   = face_recognition.face_distance(known_face_encodings, enc)
        idx     = np.argmin(dists) if dists.size else None
        name    = "Unknown"
        if idx is not None and matches[idx]:
            name = known_face_names[idx]
            mark_attendance(name)
        show_details(path, name)

    inserted_dd.bind("<<ComboboxSelected>>", on_inserted)

    Button(root, text="Random Person", command=lambda: random_person(show_details)).pack(pady=5)
    Button(root, text="Back", command=lambda:[root.destroy(), show_launcher_gui()]).pack(pady=10)

    root.mainloop()

def random_person(fn):
    if not known_face_names: return
    person = random.choice(known_face_names)
    fld = os.path.join(ORIGINAL_DIR, person)
    files = [f for f in os.listdir(fld) if f.lower().endswith(('.jpg','.png','.jpeg'))]
    if not files: return
    choice = random.choice(files)
    fn(os.path.join(fld, choice), person)

def show_loading_gui():
    root = tk.Tk()
    root.title("Loading Images")
    root.geometry("400x200")
    Label(root, text="Loading images...", font=("Arial",14)).pack(pady=20)
    pv = tk.DoubleVar()
    pb = ttk.Progressbar(root, length=300, mode='determinate', variable=pv, maximum=100)
    pb.pack(pady=10)
    lbl = Label(root, text="0%", font=("Arial",12)); lbl.pack(pady=5)
    def cb():
        root.destroy(); show_launcher_gui()
    load_face_encodings(pv, pb, lbl, cb)
    root.mainloop()

if __name__=="__main__":
    show_loading_gui()
