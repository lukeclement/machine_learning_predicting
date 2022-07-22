from math import dist
import tensorflow as tf
import glob
import os
import numpy as np


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
        for i in range(number_of_points):
            x, y = read_file("{}/boundaries_{}.dat".format(simulation, i))
            x, y = make_lines(x, y, distance)
            data = np.array([x + x_offset, y + y_offset])
            np.save("Interpolated_simulations/sim_{}_x-{}_y-{}_d-{}/{}".format(
                simulation_refs.index(simulation), x_offset, y_offset, distance, i
            ), data)
    return None


def main():
    raw_simulations = glob.glob("Simulations/*")
    valid_data = verify_data(raw_simulations)
    # Interpolate simulation datapoints
    distance_in_interpolation = 0.001
    interpolate_simulations(raw_simulations, distance=distance_in_interpolation)
    
    # Transform interpolated datapoints to images
    image_size = 128

    # Make dataset object out of images
    number_of_input_frames = 4
    timestep = 1
    
    maximum_sequence_number = 10  # How far in the sequence to look for training
    number_of_frames_in_sequence = 2  # Number of frames in the answer sequence
    # This number must be greater than 1
    # If this number is 2, then the answer sequence is just the next frame and the last frame in the sequence

    # Make neural network

    # Train network
    
    # Look at network performance

    # Extract results from network

    return 0


if __name__ == "__main__":
    main()
