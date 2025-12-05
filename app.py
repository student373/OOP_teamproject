import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
import os
import shutil
import json
import time

class DataManager:
    def __init__(self, csv_path='speciesspecies.csv'):
        self.csv_path = csv_path
        self.breed_data = None
        self.breed_features = ['Skull_Index', 'Body_Ratio', 'trainability', 'Aggression', 'Maintenance_Score']

        self.range = {
            'age' : 20.0,
            'gender': 1.0,             
            'size': 2.0,              
            'Skull_Index': 46.0,       
            'Body_Ratio': 0.7,        
            'trainability': 2.4, 
            'Aggression': 1.1,   
            'Maintenance_Score': 7.0   
        }
        
        self.feature_coefficients = {
            'age' : 10.0,
            'gender': 0.7,             
            'size': 3.0,              
            'Skull_Index': 1.0,       
            'Body_Ratio': 1.0,        
            'trainability': 1.0, 
            'Aggression': 1.0,   
            'Maintenance_Score': 1.0   
        }

        self.mismatch_penalty = 3.0 

        self.db_folder = os.path.join(os.getcwd(), 'dog_db')
        self.img_folder = os.path.join(self.db_folder, 'images')
        self.json_path = os.path.join(self.db_folder, 'dog_data.json')
        
        if not os.path.exists(self.img_folder):
            os.makedirs(self.img_folder)
            
        self.registered_dogs = [] 
        self.breed_map = {}

        self.load_breed_data()
        self.load_registered_dogs()

    def load_breed_data(self):
        if not os.path.exists(self.csv_path):
            self.breed_list = []
            print(f"경고: {self.csv_path} 파일이 없습니다.")
            return

        try:
            df = pd.read_csv(self.csv_path)
            if 'Breed' in df.columns:
                df['Breed'] = df['Breed'].str.replace('_', ' ').str.lower().str.strip()
            
            for col in self.breed_features:
                if col not in df.columns:
                    df[col] = 0
                else:
                    df[col] = df[col].fillna(0)
            
            self.breed_data = df
            self.breed_list = sorted(df['Breed'].unique().tolist())
            
            self.breed_map = df.set_index('Breed')[self.breed_features].to_dict('index')
            
        except Exception as e:
            messagebox.showerror("Error", f"견종 데이터 로드 실패: {e}")
            self.breed_list = []

    def load_registered_dogs(self):
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    self.registered_dogs = json.load(f)
            except Exception as e:
                print(f"DB 로드 오류: {e}")
                self.registered_dogs = []
        else:
            self.registered_dogs = []

    def register_dog(self, info, original_image_path):
        saved_image_path = None
        if original_image_path and os.path.exists(original_image_path):
            ext = os.path.splitext(original_image_path)[1]
            new_filename = f"{info['name']}_{int(time.time())}{ext}"
            saved_image_path = os.path.join(self.img_folder, new_filename)
            try:
                shutil.copy(original_image_path, saved_image_path)
            except Exception as e:
                print(f"이미지 복사 실패: {e}")
                saved_image_path = None
        
        info['image'] = saved_image_path
        self.registered_dogs.append(info)
        self.save_to_json()

    def save_to_json(self):
        try:
            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(self.registered_dogs, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"저장 실패: {e}")

    def calculate_matches(self, user_prefs, weights):
        if not self.registered_dogs:
            return []

        results = []

        target_breed_name = str(user_prefs.get('breed', '')).lower().strip()
        target_breed_stats = self.breed_map.get(target_breed_name, {k: 0 for k in self.breed_features})
        
        max_sq_sum = 0.0
        max_sq_sum += weights['age'] * self.feature_coefficients['age'] * (1.0 ** 2)
        max_sq_sum += weights['gender'] * self.feature_coefficients['gender'] * (1.0 ** 2)
        max_sq_sum += weights['size'] * self.feature_coefficients['size'] * (1.0 ** 2)
        
        for feature in self.breed_features:
            coeff = self.feature_coefficients.get(feature, 1.0)
            max_sq_sum += weights['breed'] * coeff * (1.0 ** 2)
        

        max_sq_sum += self.mismatch_penalty 
        
        max_distance = np.sqrt(max_sq_sum)
        if max_distance == 0: max_distance = 1.0

        for dog in self.registered_dogs:
            try:
                dog_age = float(dog.get('age', 0))
                age_diff = user_prefs['age'] - dog_age
                
                dog_gender = int(dog.get('gender', 0))
                gender_diff = user_prefs['gender'] - dog_gender
                
                dog_size = int(dog.get('size', 0))
                size_diff = user_prefs['size'] - dog_size

                dog_breed_name = str(dog.get('breed', '')).lower().strip()
                dog_breed_stats = self.breed_map.get(dog_breed_name, {k: 0 for k in self.breed_features})

                weighted_sum_sq = 0.0
                weighted_sum_sq += weights['age'] * self.feature_coefficients['age'] * ((age_diff/self.range['age']) ** 2)
                weighted_sum_sq += weights['gender'] * self.feature_coefficients['gender'] * ((gender_diff/self.range['gender']) ** 2)
                weighted_sum_sq += weights['size'] * self.feature_coefficients['size'] * ((size_diff/self.range['size']) ** 2)

                for feature in self.breed_features:
                    target_val = target_breed_stats.get(feature, 0)
                    dog_val = dog_breed_stats.get(feature, 0)
                    feat_diff = target_val - dog_val
                    
                    coeff = self.feature_coefficients.get(feature, 1.0)
                    rang = self.range.get(feature, 1.0)
                    
                    weighted_sum_sq += weights['breed'] * coeff * ((feat_diff/rang) ** 2)

                if target_breed_name != dog_breed_name:
                    weighted_sum_sq += self.mismatch_penalty

                final_distance = np.sqrt(weighted_sum_sq)
                
                ratio = final_distance / max_distance

                if ratio > 1.0: 
                    ratio = 1.0
                
                score = (1.0 - ratio) * 100
                
                results.append({
                    'dog': dog,
                    'score': round(score, 1),
                    'raw_dist': round(final_distance, 2)
                })

            except Exception as e:
                print(f"개별 강아지 계산 오류 ({dog.get('name')}): {e}")
                continue

        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results

class SearchableCombobox(ttk.Combobox):
    def __init__(self, master=None, all_values=None, **kwargs):
        super().__init__(master, **kwargs)
        self.all_values = all_values or []
        self['values'] = self.all_values
        self.bind('<KeyRelease>', self.check_input)

    def check_input(self, event):
        value = self.get()
        if value == '':
            self['values'] = self.all_values
        else:
            data = []
            for item in self.all_values:
                if value.lower() in item.lower():
                    data.append(item)
            self['values'] = data

class MainPage(tk.Frame):
    def __init__(self, master, controller):
        super().__init__(master)
        self.controller = controller
        
        label = tk.Label(self, text="유기견 매칭 시스템", font=("맑은 고딕", 24, "bold"))
        label.pack(pady=50)
        
        tk.Label(self, text=f"현재 데이터: {len(controller.data_manager.registered_dogs)}마리 로드됨", font=("맑은 고딕", 12), fg="blue").pack()
        tk.Label(self, text="데이터는 실행 폴더의 /dog_db 에 저장됩니다.", font=("맑은 고딕", 10), fg="gray").pack()

        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=20)

        btn_register = tk.Button(btn_frame, text="강아지 등록\n(Register)", width=20, height=5,
                                 command=lambda: controller.show_frame("RegisterPage"))
        btn_register.pack(side="left", padx=20)

        btn_match = tk.Button(btn_frame, text="강아지 찾기\n(Match)", width=20, height=5,
                              command=lambda: controller.show_frame("MatchPage"))
        btn_match.pack(side="right", padx=20)

class RegisterPage(tk.Frame):
    def __init__(self, master, controller):
        super().__init__(master)
        self.controller = controller
        self.image_path = None 
        
        tk.Label(self, text="강아지 등록 (Register)", font=("맑은 고딕", 18)).pack(pady=10)
        
        main_form = tk.Frame(self)
        main_form.pack(pady=10)

        form_frame = tk.Frame(main_form)
        form_frame.pack(side="left", padx=20)

        tk.Label(form_frame, text="이름:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.entry_name = tk.Entry(form_frame)
        self.entry_name.grid(row=0, column=1, padx=5, pady=5)

        tk.Label(form_frame, text="견종:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.combo_breed = SearchableCombobox(form_frame, width=27)
        self.combo_breed.grid(row=1, column=1, padx=5, pady=5)

        tk.Label(form_frame, text="나이:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.entry_age = tk.Entry(form_frame)
        self.entry_age.grid(row=2, column=1, padx=5, pady=5)

        tk.Label(form_frame, text="성별:").grid(row=3, column=0, padx=5, pady=5, sticky="e")
        self.var_gender = tk.IntVar(value=0)
        gender_frame = tk.Frame(form_frame)
        gender_frame.grid(row=3, column=1, sticky="w")
        tk.Radiobutton(gender_frame, text="수컷", variable=self.var_gender, value=0).pack(side="left")
        tk.Radiobutton(gender_frame, text="암컷", variable=self.var_gender, value=1).pack(side="left")

        tk.Label(form_frame, text="크기:").grid(row=4, column=0, padx=5, pady=5, sticky="e")
        self.combo_size = ttk.Combobox(form_frame, values=["소형", "중형", "대형"], state="readonly")
        self.combo_size.grid(row=4, column=1, padx=5, pady=5)
        self.combo_size.current(0)

        img_frame = tk.Frame(main_form)
        img_frame.pack(side="right", padx=20)

        self.lbl_preview = tk.Label(img_frame, text="이미지 없음", bg="gray", width=20, height=10)
        self.lbl_preview.pack(pady=5)

        btn_img = tk.Button(img_frame, text="사진 선택", command=self.select_image)
        btn_img.pack()

        btn_save = tk.Button(self, text="등록하기", command=self.save_dog, bg="lightblue", width=15)
        btn_save.pack(pady=20)
        
        tk.Button(self, text="뒤로가기", command=lambda: controller.show_frame("MainPage")).pack()

    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.gif")])
        if file_path:
            self.image_path = file_path
            try:
                img = Image.open(file_path)
                img = img.resize((150, 150))
                photo = ImageTk.PhotoImage(img)
                self.lbl_preview.config(image=photo, text="", width=150, height=150)
                self.lbl_preview.image = photo
            except Exception as e:
                messagebox.showerror("오류", f"이미지 로드 실패: {e}")

    def update_breeds(self):
        if self.controller.data_manager.breed_list:
            self.combo_breed.all_values = self.controller.data_manager.breed_list
            self.combo_breed['values'] = self.controller.data_manager.breed_list

    def save_dog(self):
        try:
            name = self.entry_name.get()
            breed = self.combo_breed.get()
            age_str = self.entry_age.get()
            
            if not name or not breed or not age_str:
                messagebox.showwarning("경고", "모든 정보를 입력해주세요.")
                return

            age = int(age_str)
            size_str = self.combo_size.get()
            
            size_map = {"소형": 0, "중형": 1, "대형": 2}
            
            dog_info = {
                'name': name,
                'breed': breed,
                'age': age,
                'gender': self.var_gender.get(),
                'size': size_map[size_str]
            }
            
            self.controller.data_manager.register_dog(dog_info, self.image_path)
            
            messagebox.showinfo("성공", f"{name} 등록 완료!")
            
            self.entry_name.delete(0, tk.END)
            self.entry_age.delete(0, tk.END)
            self.combo_breed.set('')
            self.lbl_preview.config(image='', text="이미지 없음", width=20, height=10)
            self.image_path = None
            
            self.controller.show_frame("MainPage")
            
        except ValueError:
            messagebox.showerror("오류", "나이는 숫자로 입력해주세요.")

class MatchPage(tk.Frame):
    def __init__(self, master, controller):
        super().__init__(master)
        self.controller = controller
        
        tk.Label(self, text="선호 조건 입력 (Preferences)", font=("맑은 고딕", 18)).pack(pady=10)
        
        container = tk.Frame(self)
        container.pack(pady=5)

        input_frame = tk.LabelFrame(container, text="조건 선택")
        input_frame.pack(side="left", padx=10, fill="y")

        tk.Label(input_frame, text="선호 나이:").grid(row=0, column=0, pady=5)
        self.entry_age = tk.Entry(input_frame, width=10)
        self.entry_age.grid(row=0, column=1, pady=5)

        tk.Label(input_frame, text="선호 성별:").grid(row=1, column=0, pady=5)
        self.var_gender = tk.IntVar(value=0)
        gf = tk.Frame(input_frame)
        gf.grid(row=1, column=1)
        tk.Radiobutton(gf, text="수컷", variable=self.var_gender, value=0).pack(side="left")
        tk.Radiobutton(gf, text="암컷", variable=self.var_gender, value=1).pack(side="left")

        tk.Label(input_frame, text="선호 크기:").grid(row=2, column=0, pady=5)
        self.combo_size = ttk.Combobox(input_frame, values=["소형", "중형", "대형"], state="readonly", width=8)
        self.combo_size.grid(row=2, column=1, pady=5)
        self.combo_size.current(0)

        tk.Label(input_frame, text="선호 견종:").grid(row=3, column=0, pady=5)
        self.combo_breed = SearchableCombobox(input_frame, width=15)
        self.combo_breed.grid(row=3, column=1, pady=5)

        weight_frame = tk.LabelFrame(container, text="중요도 (1-10)")
        weight_frame.pack(side="right", padx=10, fill="y")

        tk.Label(weight_frame, text="나이 중요도:").grid(row=0, column=0)
        self.scale_age = tk.Scale(weight_frame, from_=1, to=10, orient="horizontal", length=100)
        self.scale_age.set(5)
        self.scale_age.grid(row=0, column=1)

        tk.Label(weight_frame, text="성별 중요도:").grid(row=1, column=0)
        self.scale_gender = tk.Scale(weight_frame, from_=1, to=10, orient="horizontal", length=100)
        self.scale_gender.set(5)
        self.scale_gender.grid(row=1, column=1)

        tk.Label(weight_frame, text="크기 중요도:").grid(row=2, column=0)
        self.scale_size = tk.Scale(weight_frame, from_=1, to=10, orient="horizontal", length=100)
        self.scale_size.set(5)
        self.scale_size.grid(row=2, column=1)

        tk.Label(weight_frame, text="견종 중요도:").grid(row=3, column=0)
        self.scale_breed = tk.Scale(weight_frame, from_=1, to=10, orient="horizontal", length=100)
        self.scale_breed.set(5)
        self.scale_breed.grid(row=3, column=1)

        btn_search = tk.Button(self, text="결과 보기 (Result)", command=self.search_matches, bg="pink", width=20, height=2)
        btn_search.pack(pady=15)

        tk.Button(self, text="뒤로가기", command=lambda: controller.show_frame("MainPage")).pack()

    def update_breeds(self):
        if self.controller.data_manager.breed_list:
            self.combo_breed.all_values = self.controller.data_manager.breed_list
            self.combo_breed['values'] = self.controller.data_manager.breed_list

    def search_matches(self):
        try:
            age_val_str = self.entry_age.get()
            if not age_val_str:
                messagebox.showwarning("경고", "나이를 입력해주세요.")
                return
            age_val = int(age_val_str)
            
            breed_val = self.combo_breed.get()
            
            size_map = {"소형": 0, "중형": 1, "대형": 2}
            
            prefs = {
                'age': age_val,
                'gender': self.var_gender.get(),
                'size': size_map[self.combo_size.get()],
                'breed': breed_val
            }
            
            weights = {
                'age': self.scale_age.get(),
                'gender': self.scale_gender.get(),
                'size': self.scale_size.get(),
                'breed': self.scale_breed.get()
            }

            weights = {key: value ** 2 for key, value in weights.items()}
            
            results = self.controller.data_manager.calculate_matches(prefs, weights)
            self.controller.show_results(results)
            
        except ValueError:
            messagebox.showerror("오류", "나이는 숫자로 입력해주세요.")
        except Exception as e:
            messagebox.showerror("오류", f"검색 중 오류 발생: {e}")


class ResultPage(tk.Frame):
    def __init__(self, master, controller):
        super().__init__(master)
        self.controller = controller
        
        tk.Label(self, text="매칭 결과 (Top 50)", font=("맑은 고딕", 18)).pack(pady=10)
        
        self.canvas = tk.Canvas(self, borderwidth=0, background="#ffffff")
        self.frame = tk.Frame(self.canvas, background="#ffffff")
        self.vsb = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)

        self.vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        
        self.canvas_window = self.canvas.create_window((4,4), window=self.frame, anchor="nw", tags="self.frame")
        
        self.frame.bind("<Configure>", self.onFrameConfigure)
        self.canvas.bind("<Configure>", self.onCanvasConfigure)
        
        bottom_frame = tk.Frame(self)
        bottom_frame.pack(side="bottom", fill="x", pady=10)
        tk.Button(bottom_frame, text="메인으로", command=lambda: controller.show_frame("MainPage")).pack()

        self.photo_refs = []

    def onFrameConfigure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def onCanvasConfigure(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def display_results(self, results):
        for widget in self.frame.winfo_children():
            widget.destroy()
        self.photo_refs = []

        if not results:
            tk.Label(self.frame, text="조건에 맞는 강아지가 없습니다.", font=("맑은 고딕", 12), bg="#ffffff").pack(pady=20)
            return

        display_limit = 50
        
        for idx, item in enumerate(results[:display_limit]):
            dog = item['dog']
            score = item['score']
            
            try:
                gender_str = "암컷" if int(dog.get('gender', 0)) == 1 else "수컷"
                size_val = int(dog.get('size', 0))
                size_str = ["소형", "중형", "대형"][size_val] if 0 <= size_val <= 2 else "중형"
            except:
                gender_str = "알수없음"
                size_str = "알수없음"

            card = tk.Frame(self.frame, bd=2, relief="groove", bg="#f0f0f0")
            card.pack(fill="x", padx=10, pady=5)
            
            img_label = tk.Label(card, bg="#dddddd", width=100, height=100, text="No Image")
            img_label.pack(side="left", padx=10, pady=5)

            img_path = dog.get('image')
            if img_path and os.path.exists(img_path):
                try:
                    img = Image.open(img_path)
                    img = img.resize((100, 100))
                    photo = ImageTk.PhotoImage(img)
                    img_label.config(image=photo, text="")
                    self.photo_refs.append(photo)
                except Exception:
                    pass
            
            info_text = f"[{idx+1}위] 매칭 점수: {score}점\n\n이름: {dog.get('name', 'Unknown')}\n견종: {dog.get('breed', 'Unknown')}\n나이: {dog.get('age', '?')}살 | {gender_str} | {size_str}"
            text_label = tk.Label(card, text=info_text, justify="left", font=("맑은 고딕", 11), bg="#f0f0f0")
            text_label.pack(side="left", padx=10)

        self.frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

class DogMatchingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("유기견 매칭 시스템 (Updated)")
        self.geometry("600x650")
        
        self.data_manager = DataManager('speciesspecies.csv')

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (MainPage, RegisterPage, MatchPage, ResultPage):
            page_name = F.__name__
            frame = F(container, self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("MainPage")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        if page_name in ["RegisterPage", "MatchPage"]:
            frame.update_breeds()
        frame.tkraise()

    def show_results(self, results):
        result_page = self.frames["ResultPage"]
        result_page.display_results(results)
        self.show_frame("ResultPage")

if __name__ == "__main__":
    app = DogMatchingApp()
    app.mainloop()