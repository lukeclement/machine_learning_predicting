from math import dist
import tensorflow as tf
import glob
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def main():
    raw_simulations = glob.glob("Simulations/*")
    number_of_simulations = len(raw_simulations)
    valid_data = verify_data(raw_simulations)
    # Interpolate simulation datapoints
    distance_in_interpolation = 0.001
    # interpolate_simulations(raw_simulations, distance=distance_in_interpolation)

    # Dataset creation
    image_size = 128
    input_frames = 4
    timestep = 5

    maximum_sequence_number = 10  # How far in the sequence to look for training
    frames_in_sequence = 2  # Number of frames in the answer sequence
    # This number must be greater than 1
    # If this number is 2, then the answer sequence is just the next frame and the last frame in the sequence

    excluded_simulations = []
    number_of_sequences = get_sequence_number(number_of_simulations, distance_in_interpolation, input_frames, timestep, maximum_sequence_number, excluded_simulations)

    training_questions = np.zeros((number_of_sequences, input_frames, image_size, image_size, 1))
    training_answers = np.zeros((number_of_sequences, frames_in_sequence, image_size, image_size, 1))

    test_frame = transform_data_to_image("Interpolated_simulations/sim_0_x-0_y-0_d-0.001/0.npy", image_size)

    plt.imshow(test_frame)
    plt.show()
    # Make neural network

    # Train network

    # Look at network performance

    # Extract results from network

    return 0


def verify_data(simulation_refs):
    good_data = True
    for simulation in simulation_refs:
        dat_files = glob.glob("{}/*".format(simulation))
        number_of_points = len(dat_files)

        for i in range(number_of_points):
            if "{}/boundaries_{}.dat".format(simulation, i) not in dat_files:
                print("{}/boundaries_{}.dat".format(simulation, i))
                good_data = False
    return good_data


def read_file(file_path):
    """Takes in a filepath and extracts a set of x and y coordinates of the bubble edge.
    Input:
        A string of the file path of a .dat file to read in
    Output:
        A pair of 1D  arrays of floats (x and y)
    """
    x = []
    y = []
    try:
        file = open(file_path, "r")
        main_data = False
        for line in file:
            # Excluding data that is irrelevant (the walls of the image)
            if "boundary4" in line:
                main_data = True
            if main_data and "ZONE" not in line:
                data_points = line.strip().split(" ")
                x.append(float(data_points[0]))
                y.append(float(data_points[1]))
        file.close()
    except IOError:
        print("File {} not found.".format(file_path))
        x = []
        y = []
    except ValueError:
        print("One of the lines in {} was unexpected.".format(file_path))
    x = np.asarray(x)
    y = np.asarray(y)
    return x, y


def make_lines(x, y, resolution):
    """Creates a series of interpolated points between raw bubble edge data points.
    Inputs:
        x: A 1D array of floats from raw data points
        y: A 1D array of floats from raw data points
        resolution: A float, representing how close the interpolated points should be.
    Outputs:
        filled_x: A 1D array of interpolated data points
        filled_y: A 1D array of interpolated data points
    """
    current_x = x[0]
    current_y = y[0]
    visited = [0]
    while len(visited) < len(x):
        checked = []
        values = []
        for i in range(0, len(x)):
            if i not in visited:
                checked.append(i)
                values.append(
                    (current_x - x[i])**2 + (current_y - y[i])**2
                )
        closest = min(values)
        smallest = checked[values.index(closest)]
        visited.append(smallest)
        current_x = x[smallest]
        current_y = y[smallest]

    new_x = []
    new_y = []
    for i in visited:
        new_x.append(x[i])
        new_y.append(y[i])

    filled_x = []
    filled_y = []

    for i in range(0, len(new_x)):
        current_x = float(new_x[i])
        current_y = float(new_y[i])
        if i+1 != len(new_x):
            next_x = float(new_x[i+1])
            next_y = float(new_y[i+1])
        else:
            next_x = float(new_x[0])
            next_y = float(new_y[0])
        angle_to_next = np.arctan2(next_x - current_x, next_y - current_y)
        distance = np.sqrt((current_x - next_x)**2 + (current_y - next_y)**2)
        loops = 0
        while resolution*loops < distance:
            filled_x.append(current_x)
            filled_y.append(current_y)
            loops += 1
            current_x += resolution * np.sin(angle_to_next)
            current_y += resolution * np.cos(angle_to_next)
    filled_x = np.asarray(filled_x)
    filled_y = np.asarray(filled_y)

    return filled_x, filled_y


def interpolate_simulations(simulation_refs, distance=0.01, x_offset=0, y_offset=0):
    for simulation in simulation_refs:
        print("Working on {}".format(simulation))
        dat_files = glob.glob("{}/*".format(simulation))
        number_of_points = len(dat_files)
        try:
            os.mkdir("Interpolated_simulations/sim_{}_x-{}_y-{}_d-{}".format(
                simulation_refs.index(simulation), x_offset, y_offset, distance
            ))
        except OSError:
            print("Directory already exists")
        bar = tqdm(total=number_of_points)
        for i in range(3, number_of_points):
            x, y = read_file("{}/boundaries_{}.dat".format(simulation, i))
            x, y = make_lines(x, y, distance)
            data = np.array([x + x_offset, y + y_offset])
            np.save("Interpolated_simulations/sim_{}_x-{}_y-{}_d-{}/{}".format(
                simulation_refs.index(simulation), x_offset, y_offset, distance, i-3
            ), data)
            bar.update(1)
        bar.close()
    return None


def get_sequence_number(number_of_simulations, distance_in_interpolation, input_frames, timestep, maximum_sequence_number, excluded_simulations):
    number_of_sequences = 0
    for simulation_index in range(number_of_simulations):
        if simulation_index not in excluded_simulations:
            missing_data = False
            data_index = (input_frames + maximum_sequence_number - 1) * timestep
            while not missing_data:
                try:
                    data = np.load("Interpolated_simulations/sim_{}_x-{}_y-{}_d-{}/{}".format(
                        simulation_index, 0, 0, distance_in_interpolation, data_index
                        ))
                    number_of_sequences += 1
                except IOError:
                    missing_data = True
                data_index += 1
    return number_of_sequences


def transform_data_to_image(file_name, image_size, modifier=15):
    x, y = np.load(file_name)
    h, x_edge, y_edge = np.histogram2d(
        x, y,
        range=[[-1, 1], [-1, 1]], bins=(image_size, image_size)
    )
    output_array = np.zeros((image_size, image_size, 1))
    h = np.tanh(h/modifier)
    output_array[:, :, 0] = h.T
    return output_array


if __name__ == "__main__":
    main()
