import cv2
import glob
import os
import numpy as np
import tqdm

def process_directory_edges(input_dir, output_dir, threshold1=0, threshold2=0):
    """
    Applies Canny edge detection to all images in a directory and saves them
    as a side-by-side comparison with the original image.

    Args:
        input_dir (str): Path to the directory with input images.
        output_dir (str): Path to the directory where output images will be saved.
        threshold1 (int): First threshold for the hysteresis procedure in Canny.
        threshold2 (int): Second threshold for the hysteresis procedure in Canny.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # List of common image file extensions
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')

    # Loop through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(image_extensions):
            # Construct the full file path
            img_path = os.path.join(input_dir, filename)

            # Read the original image
            original_image = cv2.imread(img_path)
            if original_image is None:
                print(f"Warning: Could not read image '{filename}'. Skipping.")
                continue

            # 1. Convert to grayscale for edge detection
            gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

            # 2. Apply Canny edge detector
            edges = cv2.Canny(gray_image, threshold1, threshold2)

            # 3. Convert the single-channel edge image back to a 3-channel BGR image
            #    so it can be concatenated with the original color image.
            edge_image_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            # 4. Stack the original image and the edge image side-by-side
            # combined_image = np.hstack((original_image, edge_image_bgr))

            # 5. Save the resulting image
            output_path = os.path.join(output_dir, f"{filename}")
            cv2.imwrite(output_path, edge_image_bgr)
            # print(f"Processed and saved: {output_path}")


if __name__ == "__main__":
    # --- User Configuration ---
    # Define the input and output directories
    # input_directory_overall = "/home/andreww9/code/original_datasets/BP4D_croppped_MTCNN/img/"
    # output_directory_overall = "/home/andreww9/groups/grp_face_race/nobackup/autodelete/BP4D_croppped_MTCNN_edges/img/"
    input_directory_overall = "/home/andreww9/fsl_groups/grp_RVL_AU/code/DISFA_LEFT_cropped_MTCNN/img/"
    output_directory_overall = "/home/andreww9/groups/grp_face_race/nobackup/autodelete/DISFA_LEFT_cropped_MTCNN_edges/img/"
    # -------------------------

    # get person from terminal argument
    import sys
    if len(sys.argv) > 1:
        person_id = sys.argv[1]
        input_directory_people = [os.path.join(input_directory_overall, person_id)]
        print(f"Processing person ID: {person_id}")
    else:
        print("No person ID provided. Processing all persons.")
        input_directory_people = list(glob.glob(input_directory_overall + "/*"))

    for person_dir in tqdm.tqdm(input_directory_people):
        if "DISFA" in person_dir:
            input_directory_action = [person_dir]
        else:
            input_directory_action = list(glob.glob(person_dir + "/*"))
        person_name = os.path.basename(person_dir)
        for action_dir in tqdm.tqdm(input_directory_action):
            if "DISFA" in person_dir:
                this_output_dir = os.path.join(output_directory_overall, os.path.basename(action_dir))
            else:    
                this_output_dir = os.path.join(output_directory_overall, person_name, os.path.basename(action_dir))
            process_directory_edges(action_dir, this_output_dir, threshold1=0, threshold2=0)
            

    # Create a dummy input directory for demonstration if it doesn't exist
    # if not os.path.exists(input_directory):
    #     os.makedirs(input_directory)
    #     print(f"Created a sample input directory: '{input_directory}'")
    #     print("Please add some images to this folder and run the script again.")

    # process_directory_edges(input_directory, output_directory)

    # print("\nProcessing complete!")
    # print(f"Check the '{output_directory}' folder for your edge-detected images.")