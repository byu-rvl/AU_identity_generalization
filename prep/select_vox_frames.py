import cv2
import glob
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np


class MediaPipeFaceRotationEstimator:
    def __init__(self):
        model_path = '../checkpoints/face_landmarker.task'


        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            output_facial_transformation_matrixes=True,)


        self.landmarker = FaceLandmarker.create_from_options(options)

    def get_rotation_matrix(self, numpy_frame_from_opencv: np.ndarray) -> np.ndarray | None:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)
        face_landmarker_result = self.landmarker.detect(mp_image)
        
        if not face_landmarker_result.facial_transformation_matrixes or not face_landmarker_result.face_landmarks:
            print("No face detected.")
            return None, None, None
        transformation_matrix = face_landmarker_result.facial_transformation_matrixes
        landmarks = face_landmarker_result.face_landmarks
        pitch = np.arcsin(-transformation_matrix[0][2][1]) * 180.0 / np.pi
        yaw = np.arctan2(transformation_matrix[0][2][0], transformation_matrix[0][2][2]) * 180.0 / np.pi
        roll = np.arctan2(transformation_matrix[0][0][1], transformation_matrix[0][1][1]) * 180.0 / np.pi

        return transformation_matrix, [pitch, yaw, roll], landmarks

    def close(self):
        self.landmarker.close()

class find_static_faces:
    def __init__(self):
        pass

    def display_landmarks(self, image, landmarks, highlight_indices=None):
        for idx, landmark in enumerate(landmarks):
            # x, y = int(landmark[0]), int(landmark[1])
            x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
            if highlight_indices and idx in highlight_indices:
                cv2.circle(image, (x, y), 3, (0, 255, 0), -1)  # Green for highlighted points
            else:
                cv2.circle(image, (x, y), 2, (255, 0, 0), -1)  # Blue for other points
        return image

    def distance_between_points(self, point1, point2):
        point1 = [point1.x, point1.y]
        point2 = [point2.x, point2.y]
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def get_head_size(self, landmarks):
        top_center_head = landmarks[10]
        bottom_center_head = landmarks[152]
        center_head_size = self.distance_between_points(top_center_head, bottom_center_head)

        top_left_head = landmarks[338]
        bottom_left_head = landmarks[377]
        left_head_size = self.distance_between_points(top_left_head, bottom_left_head)

        top_right_head = landmarks[109]
        bottom_right_head = landmarks[148]
        right_head_size = self.distance_between_points(top_right_head, bottom_right_head)

        head_size = (center_head_size + left_head_size + right_head_size) / 3.0
        return head_size

    def distance_between_lips(self, landmarks):
        inner_center_top_lip = landmarks[13]
        inner_center_bottom_lip = landmarks[14]
        inner_center_distance = self.distance_between_points(inner_center_top_lip, inner_center_bottom_lip)
        
        outer_center_top_lip = landmarks[0]  # Top lip landmark
        outer_center_bottom_lip = landmarks[17]  # Bottom lip landmark
        outer_center_distance = self.distance_between_points(outer_center_top_lip, outer_center_bottom_lip)

        inner_left_top_lip = landmarks[312]
        inner_left_bottom_lip = landmarks[317]
        inner_left_distance = self.distance_between_points(inner_left_top_lip, inner_left_bottom_lip)

        inner_right_top_lip = landmarks[82]
        inner_right_bottom_lip = landmarks[87]
        inner_right_distance = self.distance_between_points(inner_right_top_lip, inner_right_bottom_lip)

        outer_left_top_lip = landmarks[267]
        outer_left_bottom_lip = landmarks[314]
        outer_left_distance = self.distance_between_points(outer_left_top_lip, outer_left_bottom_lip)

        outer_right_top_lip = landmarks[37]
        outer_right_bottom_lip = landmarks[84]
        outer_right_distance = self.distance_between_points(outer_right_top_lip, outer_right_bottom_lip)

        average_distance = (inner_center_distance + outer_center_distance + inner_left_distance + inner_right_distance + outer_left_distance + outer_right_distance) / 6.0

        # Normalize by the head size.
        head_size = self.get_head_size(landmarks)

        if head_size > 0:
            average_distance /= head_size

        return average_distance

    def lip_corners_relaxed(self, landmarks):
        inner_center_top_lip = landmarks[13]
        inner_center_bottom_lip = landmarks[14]
        outer_center_top_lip = landmarks[0]
        outer_center_bottom_lip = landmarks[17]
        inner_left_top_lip = landmarks[312]
        inner_left_bottom_lip = landmarks[317]
        inner_right_top_lip = landmarks[82]
        inner_right_bottom_lip = landmarks[87]
        outer_left_top_lip = landmarks[267]
        outer_left_bottom_lip = landmarks[314]
        outer_right_top_lip = landmarks[37]
        outer_right_bottom_lip = landmarks[84]

        mouth_center_y = (inner_center_top_lip.y + inner_center_bottom_lip.y + outer_center_top_lip.y + outer_center_bottom_lip.y + 
                         inner_left_top_lip.y + inner_left_bottom_lip.y + inner_right_top_lip.y + inner_right_bottom_lip.y + 
                         outer_left_top_lip.y + outer_left_bottom_lip.y + outer_right_top_lip.y + outer_right_bottom_lip.y) / 12.0

        mouth_center_x = (inner_center_top_lip.x + inner_center_bottom_lip.x + outer_center_top_lip.x + outer_center_bottom_lip.x + 
                         inner_left_top_lip.x + inner_left_bottom_lip.x + inner_right_top_lip.x + inner_right_bottom_lip.x + 
                         outer_left_top_lip.x + outer_left_bottom_lip.x + outer_right_top_lip.x + outer_right_bottom_lip.x) / 12.0

        right_outer_corner = landmarks[61]
        left_outer_corner = landmarks[291]

        right_inner_corner = landmarks[78]
        left_inner_corner = landmarks[308]

        corner_average_y = (right_outer_corner.y + left_outer_corner.y + right_inner_corner.y + left_inner_corner.y) / 4.0
        corner_right_average_x = (right_outer_corner.x + right_inner_corner.x) / 2.0
        corner_left_average_x = (left_outer_corner.x + left_inner_corner.x) / 2.0

        y_distance = abs(mouth_center_y - corner_average_y)
        # X distance is how far apart the corners are relative to the mouth center
        x_distance = abs(abs(corner_right_average_x - mouth_center_x) - abs(corner_left_average_x - mouth_center_x))

        # Normalize by the head size.
        head_size = self.get_head_size(landmarks)
        if head_size > 0:
            y_distance /= head_size
            x_distance /= head_size

        return x_distance + y_distance

    def eye_open_amnt(self, landmarks):
        left_eye_center_top = landmarks[386]
        left_eye_center_bottom = landmarks[374]
        left_eye_center_distance = abs(left_eye_center_top.y - left_eye_center_bottom.y)

        left_eye_in_top = landmarks[385]
        left_eye_in_bottom = landmarks[380]
        left_eye_in_distance = abs(left_eye_in_top.y - left_eye_in_bottom.y)

        left_eye_out_top = landmarks[387]
        left_eye_out_bottom = landmarks[373]
        left_eye_out_distance = abs(left_eye_out_top.y - left_eye_out_bottom.y)

        left_eye_distance = (left_eye_center_distance + left_eye_in_distance + left_eye_out_distance) / 3.0

        right_eye_center_top = landmarks[159]
        right_eye_center_bottom = landmarks[145]
        right_eye_center_distance = abs(right_eye_center_top.y - right_eye_center_bottom.y)

        right_eye_in_top = landmarks[158]
        right_eye_in_bottom = landmarks[153]
        right_eye_in_distance = abs(right_eye_in_top.y - right_eye_in_bottom.y)

        right_eye_out_top = landmarks[160]
        right_eye_out_bottom = landmarks[144]
        right_eye_out_distance = abs(right_eye_out_top.y - right_eye_out_bottom.y)

        right_eye_distance = (right_eye_center_distance + right_eye_in_distance + right_eye_out_distance) / 3.0

        average_eye_distance = (left_eye_distance + right_eye_distance) / 2.0

        # Normalize by the head size.
        head_size = self.get_head_size(landmarks)
        if head_size > 0:
            average_eye_distance /= head_size

        return average_eye_distance

    def find_best_frame_list(self, list_of_frames, frame_names = []):
        '''Rank each of them with lowest number being best, then add 
        them. The one with the lowest number wins. If there is a tie, 
        pick the top priority method and do the one that scored the 
        best rank there.
        '''

        results = {}
        all_rolls = []
        all_pitches = []
        all_yaws = []
        all_lip_distances = []
        all_amnt_unrelaxed = []
        all_eye_open_amounts = []

        estimator = MediaPipeFaceRotationEstimator()

        if len(frame_names) == 0:
            frame_names = [f"frame_{i}" for i in range(len(list_of_frames))]

        success_frame_cntr = 0

        for frame_name, frame in zip(frame_names, list_of_frames):
            rot_matrix, euler_angles, landmarks = estimator.get_rotation_matrix(frame)
            if not landmarks:
                continue
            
            for i, landmark in enumerate(landmarks):
                if len(landmark) == 0:
                    continue
                lip_distance = self.distance_between_lips(landmark)
                amnt_unrelaxed = self.lip_corners_relaxed(landmark)
                eye_open_amount = self.eye_open_amnt(landmark)
                all_lip_distances.append(lip_distance)
                all_amnt_unrelaxed.append(amnt_unrelaxed)
                all_eye_open_amounts.append(eye_open_amount)
                all_rolls.append(abs(euler_angles[2]))
                all_pitches.append(abs(euler_angles[0]))
                all_yaws.append(abs(euler_angles[1]))

                results[f'{frame_name} face{i}_of_{len(landmarks)}'] = (lip_distance, amnt_unrelaxed, eye_open_amount, euler_angles)

            success_frame_cntr += 1
        estimator.close()

        all_lip_distances = np.array(all_lip_distances)
        all_amnt_unrelaxed = np.array(all_amnt_unrelaxed)
        all_eye_open_amounts = np.array(all_eye_open_amounts)
        all_rolls = np.array(all_rolls)
        all_pitches = np.array(all_pitches)
        all_yaws = np.array(all_yaws)

        lip_distance_ranks = np.argsort(np.argsort(all_lip_distances))
        amnt_unrelaxed_ranks = np.argsort(np.argsort(all_amnt_unrelaxed))
        eye_open_amount_ranks = np.argsort(np.argsort(-all_eye_open_amounts))  # Negative for descending order
        roll_ranks = np.argsort(np.argsort(all_rolls))
        pitch_ranks = np.argsort(np.argsort(all_pitches))
        yaw_ranks = np.argsort(np.argsort(all_yaws))
        final_scores = lip_distance_ranks + amnt_unrelaxed_ranks + eye_open_amount_ranks + roll_ranks + pitch_ranks + yaw_ranks

        best_index = np.argmin(final_scores)
        best_frame_name = list(results.keys())[best_index]
        best_metrics = results[best_frame_name]

        print("Best frame:", best_frame_name)
        print("Metrics (lip_distance, amnt_unrelaxed, eye_open_amount):", best_metrics)

        # Print all results in order of final score
        sorted_indices = np.argsort(final_scores)
        print("\nAll frames ranked by final score:")
        for rank, idx in enumerate(sorted_indices):
            frame_name = list(results.keys())[idx]
            metrics = results[frame_name]
            print(f"Rank {rank + 1}: {frame_name} - Metrics: {metrics}")
            # Only print the first 200
            if rank >= 200:
                break

        print("Successfully processed frames percentage:", (success_frame_cntr / len(frame_names)) * 100, len(frame_names), success_frame_cntr)

        
        return best_frame_name, best_metrics
        

if __name__ == '__main__':
    image_directory = "/home/andreww9/groups/grp_face_race/code/vox_celeb_identities/"
    all_images = []
    all_image_names = []
    for img_path in glob.glob(image_directory + "*.jpg"):
        img = cv2.imread(img_path)
        if img is not None:
            all_images.append(img)
            all_image_names.append(img_path)
    analysis = find_static_faces()
    analysis.find_best_frame_list(all_images, all_image_names)



    # Make sure you have an image file named 'face_image.jpg' in the same directory,
    # or change the path to your image file.
    # image_path = '/home/andreww9/groups/grp_face_race/code/vox_celeb_identities/id00019_frame.jpg'
    # # image_path = '/home/andreww9/groups/grp_face_race/code/vox_celeb_identities/id00026_frame.jpg'
    # # image_path = '/home/andreww9/groups/grp_face_race/code/vox_celeb_identities/id00078_frame.jpg'
    # # image_path = '/home/andreww9/groups/grp_face_race/code/vox_celeb_identities/id00188_frame.jpg'
    
    # input_image = cv2.imread(image_path)
    # if input_image is None:
    #     raise FileNotFoundError(f"Image not found at path: {image_path}")

    # # 1. Initialize
    # estimator = MediaPipeFaceRotationEstimator()
    # analysis = find_static_faces()

    # # 3. Clean up
    # estimator.close()

    # # 4. Find important information from landmarks
    # if landmarks:
    #     # 2. Get the rotation matrix from the image
    #     rot_matrix, euler_angles, landmarks = estimator.get_rotation_matrix(input_image)
    #     print(f"Euler angles (pitch, yaw, roll): {euler_angles}")
    #     lip_distance = analysis.distance_between_lips(landmarks[0])
    #     print(f"Average distance between lips: {lip_distance}")
    #     amnt_unrelaxed = analysis.lip_corners_relaxed(landmarks[0])
    #     print(f"Lip corners relaxed distances - amnt_unrelaxed: {amnt_unrelaxed}")
    #     eye_open_amount = analysis.eye_open_amnt(landmarks[0])
    #     print(f"Average eye open amount: {eye_open_amount}")

    # print landmarks:
    # cv2.imwrite('landmarks_inner_center_lip.jpg', analysis.display_landmarks(input_image.copy(), landmarks[0], highlight_indices=[13, 14]))
    # cv2.imwrite('landmarks_outer_center_lip.jpg', analysis.display_landmarks(input_image.copy(), landmarks[0], highlight_indices=[0, 17]))
    # cv2.imwrite('landmarks_inner_left_lip.jpg', analysis.display_landmarks(input_image.copy(), landmarks[0], highlight_indices=[312, 317]))
    # cv2.imwrite('landmarks_inner_right_lip.jpg', analysis.display_landmarks(input_image.copy(), landmarks[0], highlight_indices=[82, 87]))
    # cv2.imwrite('landmarks_outer_left_lip.jpg', analysis.display_landmarks(input_image.copy(), landmarks[0], highlight_indices=[267, 314]))
    # cv2.imwrite('landmarks_outer_right_lip.jpg', analysis.display_landmarks(input_image.copy(), landmarks[0], highlight_indices=[37, 84]))
    # cv2.imwrite('landmarks_head_center.jpg', analysis.display_landmarks(input_image.copy(), landmarks[0], highlight_indices=[10, 152]))
    # cv2.imwrite('landmarks_head_left.jpg', analysis.display_landmarks(input_image.copy(), landmarks[0], highlight_indices=[338, 377]))
    # cv2.imwrite('landmarks_head_right.jpg', analysis.display_landmarks(input_image.copy(), landmarks[0], highlight_indices=[109, 148]))
    # cv2.imwrite('lip_corners_inner.jpg', analysis.display_landmarks(input_image.copy(), landmarks[0], highlight_indices=[61, 291]))
    # cv2.imwrite('lip_corners_outer.jpg', analysis.display_landmarks(input_image.copy(), landmarks[0], highlight_indices=[78, 308]))