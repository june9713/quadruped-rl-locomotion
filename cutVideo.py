import os
from moviepy.editor import VideoFileClip
import time

# --- ⚠️ 설정 (사용 전 반드시 수정해주세요) ---

# 1. 원본 동영상 파일들이 있는 폴더 경로
# 예: "C:/Users/YourName/Videos/Originals"
INPUT_FOLDER = "D:/workspace/python/2025/quadruped-rl-locomotion/training_videos"

# 2. 편집된 동영상들을 저장할 폴더 경로
# 예: "C:/Users/YourName/Videos/Trimmed"
# 원본 보호를 위해 반드시 다른 폴더로 지정해주세요.
OUTPUT_FOLDER = "D:/workspace/python/2025/quadruped-rl-locomotion/training_videos/trimmed"


# 3. 각 동영상 앞에서 잘라낼 시간 (초 단위)
# 예: 5초를 잘라내려면 5, 1분 30초를 잘라내려면 90
CUT_DURATION_SECONDS = 10

# --- 스크립트 본문 (수정할 필요 없음) ---

def trim_videos_in_folder(input_dir, output_dir, cut_from_start):
    """
    지정된 폴더의 모든 mp4 동영상 앞부분을 잘라내고 다른 폴더에 저장합니다.
    """
    # 결과물 저장 폴더가 없으면 생성
    if not os.path.exists(output_dir):
        print(f"'{output_dir}' 폴더가 없어 새로 생성합니다.")
        os.makedirs(output_dir)

    # 입력 폴더에서 파일 목록 가져오기
    try:
        files = os.listdir(input_dir)
    except FileNotFoundError:
        print(f"오류: 입력 폴더 '{input_dir}'를 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    # mp4 파일만 필터링
    mp4_files = [f for f in files if f.lower().endswith(".mp4")]

    if not mp4_files:
        print(f"'{input_dir}' 폴더에 처리할 .mp4 파일이 없습니다.")
        return

    total_files = len(mp4_files)
    print(f"총 {total_files}개의 .mp4 파일을 처리합니다.")
    print("-" * 30)

    for i, filename in enumerate(mp4_files):
        start_time = time.time()
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"trimmed_{filename}")

        print(f"({i+1}/{total_files}) 처리 중: {filename}")

        # 이미 처리된 파일이 있다면 건너뛰기
        if os.path.exists(output_path):
            print("-> 이미 처리된 파일이므로 건너뜁니다.")
            continue

        try:
            # with 구문을 사용하여 파일을 열고 자동으로 닫도록 처리
            with VideoFileClip(input_path) as video:
                # 동영상 총 길이가 잘라낼 시간보다 짧은 경우 건너뛰기
                if video.duration <= cut_from_start:
                    print(f"-> 동영상 길이({video.duration:.2f}초)가 잘라낼 시간({cut_from_start}초)보다 짧아 건너뜁니다.")
                    continue
                
                # subclip을 사용하여 동영상 자르기
                # 예: subclip(5) -> 5초부터 끝까지
                trimmed_clip = video.subclip(cut_from_start)

                # 잘라낸 동영상을 파일로 저장
                # codec은 호환성을 위해 지정하는 것이 좋습니다.
                trimmed_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
            
            end_time = time.time()
            print(f"-> 완료! 저장 경로: {output_path} (소요 시간: {end_time - start_time:.2f}초)")

        except Exception as e:
            print(f"-> 처리 중 오류 발생: {filename}, 오류: {e}")
        
        print("-" * 30)

    print("모든 작업이 완료되었습니다.")


if __name__ == "__main__":
    trim_videos_in_folder(INPUT_FOLDER, OUTPUT_FOLDER, CUT_DURATION_SECONDS)