import cv2
import os
import glob

def create_timelapse_opencv(input_folder, output_file, speed_factor=4, fps=30):
    """
    OpenCV를 사용하여 여러 MP4 파일을 합치고 타임랩스 생성
    
    Args:
        input_folder (str): MP4 파일들이 있는 폴더 경로
        output_file (str): 출력 파일명
        speed_factor (int): 재생속도 배율 (몇 프레임마다 하나씩 선택할지)
        fps (int): 출력 비디오의 FPS
    """
    
    # MP4 파일 목록 가져오기
    video_files = sorted(glob.glob(os.path.join(input_folder, "*.mp4")))
    video_files.sort(key=lambda x: os.path.getctime(x))  # 생성 시간 기준 정렬
    
    if not video_files:
        print("MP4 파일을 찾을 수 없습니다.")
        return
    
    print(f"찾은 파일 수: {len(video_files)}")
    
    # 첫 번째 비디오에서 해상도 정보 가져오기
    first_video = cv2.VideoCapture(video_files[0])
    width = int(first_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(first_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first_video.release()
    
    # 출력 비디오 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    frame_count = 0
    
    # 각 비디오 파일 처리
    for video_file in video_files:
        print(f"처리 중: {os.path.basename(video_file)}")
        cap = cv2.VideoCapture(video_file)
        
        local_frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # speed_factor마다 하나의 프레임만 선택 (타임랩스 효과)
            if local_frame_count % speed_factor == 0:
                # 해상도가 다르면 리사이즈
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))
                
                out.write(frame)
                frame_count += 1
            
            local_frame_count += 1
        
        cap.release()
    
    out.release()
    print(f"타임랩스 영상이 완성되었습니다: {output_file}")
    print(f"총 프레임 수: {frame_count}")

def create_timelapse_with_files_opencv(file_list, output_file, speed_factor=4, fps=30):
    """
    파일 리스트를 직접 지정하여 OpenCV로 타임랩스 생성
    """
    if not file_list:
        print("파일 리스트가 비어있습니다.")
        return
    
    # 첫 번째 비디오에서 해상도 가져오기
    first_video = cv2.VideoCapture(file_list[0])
    width = int(first_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(first_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first_video.release()
    
    # 출력 비디오 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    frame_count = 0
    
    for video_file in file_list:
        if not os.path.exists(video_file):
            print(f"파일을 찾을 수 없습니다: {video_file}")
            continue
            
        print(f"처리 중: {os.path.basename(video_file)}")
        cap = cv2.VideoCapture(video_file)
        
        local_frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if local_frame_count % speed_factor == 0:
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))
                
                out.write(frame)
                frame_count += 1
            
            local_frame_count += 1
        
        cap.release()
    
    out.release()
    print(f"타임랩스 영상이 완성되었습니다: {output_file}")

# 사용 예시
if __name__ == "__main__":
    # 방법 1: 폴더의 모든 MP4 파일 처리
    input_folder = "./eval_videos_standing"
    output_file = "timelapse_opencv.mp4"
    speed_factor = 4  # 4프레임마다 1프레임 선택 (4배속 효과)
    fps = 30  # 출력 비디오 FPS
    
    create_timelapse_opencv(input_folder, output_file, speed_factor, fps)
    
    # 방법 2: 특정 파일들만 처리
    # file_list = ["video1.mp4", "video2.mp4", "video3.mp4"]
    # create_timelapse_with_files_opencv(file_list, "custom_timelapse.mp4", 6, 25)