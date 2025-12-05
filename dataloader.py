import os
import shutil
import json
import random
import time
import kagglehub
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FOLDER = os.path.join(BASE_DIR, 'dog_db')
DB_IMG_FOLDER = os.path.join(DB_FOLDER, 'images')
DB_JSON_PATH = os.path.join(DB_FOLDER, 'dog_data.json')

MAX_DOGS_PER_BREED = 10

MAX_WORKERS = 32

DOG_NAMES = [
    "Buddy", "Bella", "Charlie", "Lucy", "Max", "Luna", "Bailey", "Daisy", 
    "Cooper", "Coco", "Rocky", "Molly", "Bear", "Maggie", "Duke", "Sophie",
    "Teddy", "Chloe", "Toby", "Sadie", "Jack", "Lola", "Oliver", "Stella"
]

def clean_breed_name(folder_name):
    try:
        parts = folder_name.split('-', 1)
        if len(parts) > 1:
            return parts[1].replace('_', ' ').lower().strip()
        return folder_name.replace('_', ' ').lower().strip()
    except Exception:
        return "unknown"

def process_single_image(args):
    src_path, dest_folder, breed_name = args
    
    try:
        name = random.choice(DOG_NAMES)
        age = int(random.randint(1, 15))
        gender = int(random.randint(0, 1))
        size = int(random.randint(0, 2))

        ext = os.path.splitext(src_path)[1]
        safe_breed = breed_name.replace(' ', '')
        new_filename = f"{safe_breed}_{int(time.time())}_{random.randint(1000,9999)}{ext}"
        
        dst_path = os.path.join(dest_folder, new_filename)
        
        shutil.copy2(src_path, dst_path)
        
        return {
            'name': name,
            'breed': breed_name,
            'age': age,
            'gender': gender,
            'size': size,
            'image': dst_path
        }
    except Exception as e:
        return None

def main():
    print(">>> 1. Kaggle Dataset 다운로드/확인 시작...")
    try:
        source_root = kagglehub.dataset_download("jessicali9530/stanford-dogs-dataset")
        print(f"Dataset location: {source_root}")
    except Exception as e:
        print(f"다운로드 실패: {e}")
        return

    print("\n>>> 2. 기존 DB 폴더 정리 중...")
    if os.path.exists(DB_FOLDER):
        try:
            shutil.rmtree(DB_FOLDER)
            time.sleep(1) # 삭제 대기
        except Exception as e:
            print(f"폴더 삭제 중 경고: {e}")
            
    os.makedirs(DB_IMG_FOLDER, exist_ok=True)
    print(f"생성된 DB 폴더: {DB_FOLDER}")

    print(">>> 3. 작업 목록 생성 중...")
    tasks = []
    
    for root, dirs, files in os.walk(source_root):
        folder_name = os.path.basename(root)
        
        if folder_name.startswith('n0'):
            breed_name = clean_breed_name(folder_name)
            
            image_files = [
                os.path.join(root, f) 
                for f in files 
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            
            if not image_files:
                continue

            random.shuffle(image_files)
            selected_files = image_files[:MAX_DOGS_PER_BREED]
            
            for src_path in selected_files:
                tasks.append((src_path, DB_IMG_FOLDER, breed_name))

    if not tasks:
        print("오류: 이미지 파일을 찾을 수 없습니다.")
        return

    print(f"총 {len(tasks)}개의 이미지 처리 예정. (스레드: {MAX_WORKERS})")
    print(">>> 4. 데이터 생성 및 파일 복사 시작...")

    registered_dogs = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {executor.submit(process_single_image, task): task for task in tasks}
        
        completed = 0
        total = len(tasks)
        
        for future in as_completed(future_to_task):
            result = future.result()
            if result:
                registered_dogs.append(result)
            
            completed += 1
            if completed % 100 == 0 or completed == total:
                print(f"\r진행률: {completed}/{total} ({(completed/total)*100:.1f}%)", end='')

    print(f"\n완료! 소요 시간: {time.time() - start_time:.2f}초")

    print(f">>> 5. JSON 데이터 저장 중 ({len(registered_dogs)}건)...")
    with open(DB_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(registered_dogs, f, ensure_ascii=False, indent=4)

    print(">>> 6. 원본 캐시 데이터 정리 중...")
    try:
        shutil.rmtree(source_root)
        print("원본 데이터 삭제 완료.")
    except Exception as e:
        print(f"원본 삭제 실패 (수동 삭제 요망): {e}")

    print("\n" + "="*40)
    print(">>> [데이터 검증]")
    if registered_dogs:
        print("생성된 데이터 예시 (1건):")
        print(json.dumps(registered_dogs[0], indent=2, ensure_ascii=False))
        print(f"데이터 파일 위치: {DB_JSON_PATH}")
    else:
        print("경고: 생성된 데이터가 없습니다!")
    print("="*40)
    print("이제 app.py를 실행하세요.")

if __name__ == "__main__":
    main()