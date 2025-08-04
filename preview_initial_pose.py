import os
import time
import matplotlib.pyplot as plt
from go1_standing_env import BipedalWalkingEnv

def preview_and_save_initial_pose():
    """
    BipedalWalkingEnv의 초기 자세를 시각적으로 미리보고 이미지 파일로 저장합니다.
    """
    print("🐾 2족 보행 환경의 초기 자세를 생성합니다...")

    # 1. 렌더링 모드를 'rgb_array'로 설정하여 환경을 생성합니다.
    # 이 모드는 시뮬레이션 화면을 이미지 데이터(numpy 배열)로 반환해줍니다.
    env = BipedalWalkingEnv(render_mode="rgb_array")

    # 2. env.reset()을 호출합니다.
    # 이 과정에서 BipedalWalkingEnv가 상속받은 Go1StandingEnv의
    # _set_bipedal_ready_pose() 함수가 자동으로 실행되어 로봇을 초기 자세로 설정합니다.
    env.reset()

    # 3. env.render()를 호출하여 현재 시점의 이미지를 가져옵니다.
    image_data = env.render()

    # 4. 환경을 종료합니다.
    env.close()

    if image_data is None:
        print("❌ 이미지를 렌더링하는 데 실패했습니다. 렌더링 설정을 확인하세요.")
        return

    print("🖼️ 이미지를 생성했습니다. 화면에 표시하고 파일로 저장합니다.")

    # 5. Matplotlib를 사용하여 이미지를 화면에 표시합니다.
    plt.figure(figsize=(8, 6))
    plt.imshow(image_data)
    plt.title("로봇 초기 자세 미리보기 (Initial Pose Preview)")
    plt.axis('off')  # 축 정보는 숨깁니다.
    plt.show()

    # 6. 생성된 이미지를 파일로 저장합니다.
    # 'previews'라는 폴더를 만들고 그 안에 저장합니다.
    output_dir = "previews"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = int(time.time())
    filename = os.path.join(output_dir, f"initial_pose_{timestamp}.png")
    
    # Matplotlib의 imsave를 사용하여 numpy 배열을 이미지 파일로 직접 저장
    plt.imsave(filename, image_data)
    
    print(f"✅ 초기 자세 이미지가 성공적으로 저장되었습니다: {filename}")


if __name__ == "__main__":
    preview_and_save_initial_pose()