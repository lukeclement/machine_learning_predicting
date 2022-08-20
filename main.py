from time import time
import imageio
from math import dist
import tensorflow as tf
import glob
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.keras import layers, models, initializers, losses, optimizers, activations, metrics, Input, Model
from tensorflow.keras import backend as k

def main():
    raw_simulations = glob.glob("Simulations/*")
    number_of_simulations = len(raw_simulations)
    valid_data = verify_data(raw_simulations)
    # Interpolate simulation datapoints
    distance_in_interpolation = 0.001
    # interpolate_simulations(raw_simulations, distance=distance_in_interpolation)

    # Dataset creation for images
    batch_size = 64
    image_size = 128
    input_frames = 4
    timestep = 5

    maximum_sequence_number = 10  # How far in the sequence to look for training
    frames_in_sequence = 2  # Number of frames in the answer sequence
    # This number must be greater than 1
    # If this number is 2,
    # then the answer sequence is just the next frame and the last frame in the sequence

    excluded_simulations = []
    number_of_sequences = get_sequence_number(number_of_simulations, distance_in_interpolation,
                                              input_frames, timestep, maximum_sequence_number,
                                              excluded_simulations)

    training_questions = np.zeros((
        number_of_sequences, input_frames, image_size, image_size, 1
    ))
    training_answers = np.zeros((
        number_of_sequences, frames_in_sequence, image_size, image_size, 1
    ))

    print(number_of_sequences)
    pbar = tqdm(total=number_of_sequences)
    sequence_index = 0
    for simulation_index in range(number_of_simulations):
        if simulation_index not in excluded_simulations:
            number_of_data_points = len(glob.glob(
                "Interpolated_simulations/sim_{}_x-{}_y-{}_d-{}/*".format(
                    simulation_index, 0, 0, distance_in_interpolation
                )
            ))
            max_data_index = number_of_data_points - (maximum_sequence_number + input_frames - 1)*timestep
            for data_index in range(max_data_index):
                first_index = data_index
                for frame_index in range(input_frames):
                    data_point = first_index + frame_index * timestep
                    filename = "Interpolated_simulations/sim_{}_x-{}_y-{}_d-{}/{}.npy".format(
                        simulation_index, 0, 0, distance_in_interpolation, data_point
                    )
                    training_questions[
                        sequence_index, frame_index, :, :, :
                    ] = transform_data_to_image(filename, image_size)

                for next_stages in range(frames_in_sequence):
                    data_point = first_index + (
                        input_frames +
                        next_stages * (maximum_sequence_number-1)//(frames_in_sequence-1)
                    ) * timestep
                    filename = "Interpolated_simulations/sim_{}_x-{}_y-{}_d-{}/{}.npy".format(
                        simulation_index, 0, 0, distance_in_interpolation, data_point
                    )
                    training_answers[
                        sequence_index, next_stages, :, :, :
                    ] = transform_data_to_image(filename, image_size)
                sequence_index += 1
                pbar.update(1)
    pbar.close()

    testing_data = tf.data.Dataset.from_tensor_slices((training_questions, training_answers))
    testing_data = testing_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    # Make neural network
    generator = create_basic_network(layers.LeakyReLU(), input_frames, image_size, channels=1)
    discriminator = create_discriminator(input_frames, image_size)
    lr_schedule = LearningRateStep(1e-3)
    generator_optimizer = optimizers.Adam(learning_rate=lr_schedule, epsilon=0.1)
    discriminator_optimizer = optimizers.Adam(learning_rate=lr_schedule, epsilon=0.1)

    # Train network

    # Look at network performance

    # Extract results from network

    return 0


class LearningRateStep(optimizers.schedules.LearningRateSchedule):
    def __init__(self, i_lr):
        self.initial_learning_rate = i_lr

    def __call__(self, step):
        epoch = step//1296
        return self.initial_learning_rate * 0.1 ** (epoch//20)


def create_basic_network(activation, input_frames, image_size, channels=3, latent_dimensions=100, start_channels=32):
    input_layer = layers.Input(shape=(input_frames, image_size, image_size, channels))
    layer_depth = -1
    frames = input_frames
    size = image_size
    x = input_layer
    frames_ran = False
    while max(frames, 1) * size * size > latent_dimensions:
        layer_depth += 1
        if frames > 1:
            x = layers.Conv3D(start_channels*2**layer_depth, 5, strides=2, padding='same')(x)
            x = activation(x)
            frames = frames // 2
            size = size // 2
            frames_ran = True
        else:
            if frames == 1:
                if frames_ran:
                    x = layers.Reshape((size, size, int(start_channels*2**(layer_depth-1))))(x)
                else:
                    x = layers.Reshape((size, size, channels))(x)
                frames = 0
            x = layers.Conv2D(start_channels*2**layer_depth, 5, strides=2, padding='same', use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = activation(x)
            size = size // 2

    # x = layers.Flatten()(x)
    # x = layers.Dense(8 * 8 * 32 * 2**layer_depth)(x)
    # x = layers.BatchNormalization()(x)
    # x = activation(x)
    # x = layers.Reshape((8, 8, 32 * 2**layer_depth))(x)
    while size != image_size:
        layer_depth -= 1
        x = layers.Conv2DTranspose(start_channels*2**layer_depth, 5, strides=2, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = activation(x)
        size = size * 2

    x = layers.Conv2DTranspose(1, 5, activation='sigmoid', padding='same', use_bias=False)(x)

    model = Model(input_layer, x)
    return model


def create_discriminator(input_frames, input_size):
    input_layer = Input(shape=(input_frames, input_size, input_size, 1))
    x = layers.Conv3D(32, 3, strides=2, padding='same')(input_layer)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv3D(64, 3, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv3D(128, 3, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    x = layers.Activation('sigmoid', dtype='float32')(x)

    model = Model(input_layer, x, name='discriminator')
    return model


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
        number_of_points = len(dat_files)-5
        try:
            os.mkdir("Interpolated_simulations/sim_{}_x-{}_y-{}_d-{}".format(
                simulation_refs.index(simulation), x_offset, y_offset, distance
            ))
        except OSError:
            print("Directory already exists")
        bar = tqdm(total=number_of_points-3)
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


def get_sequence_number(number_of_simulations, distance_in_interpolation, input_frames,
                        timestep, maximum_sequence_number, excluded_simulations):
    number_of_sequences = 0
    for simulation_index in range(number_of_simulations):
        if simulation_index not in excluded_simulations:
            missing_data = False
            data_index = (input_frames + maximum_sequence_number - 1) * timestep
            while not missing_data:
                try:
                    data = np.load("Interpolated_simulations/sim_{}_x-{}_y-{}_d-{}/{}.npy".format(
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
