import os
import shutil
import json
import random
import time
import kagglehub
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 설정값 ---
# app.py와 정확히 일치하는 경로 설정 (절대 경로 사용)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FOLDER = os.path.join(BASE_DIR, 'dog_db')
DB_IMG_FOLDER = os.path.join(DB_FOLDER, 'images')
DB_JSON_PATH = os.path.join(DB_FOLDER, 'dog_data.json')

# 견종당 최대 등록 마리 수 (테스트를 위해 전체 다 하려면 1000 유지)
MAX_DOGS_PER_BREED = 1000

# 병렬 처리 스레드 수
MAX_WORKERS = 16

DOG_NAMES = [
    "Buddy", "Bella", "Charlie", "Lucy", "Max", "Luna", "Bailey", "Daisy", 
    "Cooper", "Coco", "Rocky", "Molly", "Bear", "Maggie", "Duke", "Sophie",
    "Teddy", "Chloe", "Toby", "Sadie", "Jack", "Lola", "Oliver", "Stella"
]

def clean_breed_name(folder_name):
    """
    폴더명에서 견종 이름 정제
    예: 'n02099601-golden_retriever' -> 'Golden Retriever'
    """
    try:
        parts = folder_name.split('-', 1)
        if len(parts) > 1:
            # 언더바를 공백으로 바꾸고 타이틀 케이스로 변환
            return parts[1].replace('_', ' ').title().strip()
        return folder_name.replace('_', ' ').title().strip()
    except Exception:
        return "Unknown"

def process_single_image(args):
    """단일 이미지 처리 및 데이터 생성"""
    src_path, dest_folder, breed_name = args
    
    try:
        # 랜덤 데이터 생성 (app.py의 데이터 타입과 정확히 일치시킴)
        name = random.choice(DOG_NAMES)
        age = int(random.randint(1, 15))      # int 명시
        gender = int(random.randint(0, 1))    # int 명시 (0: 수컷, 1: 암컷)
        size = int(random.randint(0, 2))      # int 명시 (0: 소, 1: 중, 2: 대)
        
        # 파일명 생성: 견종_시간_난수.ext
        ext = os.path.splitext(src_path)[1]
        safe_breed = breed_name.replace(' ', '')
        new_filename = f"{safe_breed}_{int(time.time())}_{random.randint(1000,9999)}{ext}"
        
        # app.py가 이미지를 로드할 때 절대 경로를 선호하므로 절대 경로 생성
        dst_path = os.path.join(dest_folder, new_filename)
        
        # 파일 복사
        shutil.copy2(src_path, dst_path)
        
        # 메타데이터 반환
        return {
            'name': name,
            'breed': breed_name,
            'age': age,
            'gender': gender,
            'size': size,
            'image': dst_path  # 절대 경로 저장
        }
    except Exception as e:
        # 파일 손상 등으로 실패 시 무시
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
    
    # os.walk로 모든 하위 폴더 탐색
    for root, dirs, files in os.walk(source_root):
        folder_name = os.path.basename(root)
        
        # 'n0...'으로 시작하는 견종 폴더만 대상
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

    # 5. JSON 저장
    print(f">>> 5. JSON 데이터 저장 중 ({len(registered_dogs)}건)...")
    with open(DB_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(registered_dogs, f, ensure_ascii=False, indent=4)

    # 6. 원본 삭제
    print(">>> 6. 원본 캐시 데이터 정리 중...")
    try:
        shutil.rmtree(source_root)
        print("원본 데이터 삭제 완료.")
    except Exception as e:
        print(f"원본 삭제 실패 (수동 삭제 요망): {e}")

    # 7. 검증 출력
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