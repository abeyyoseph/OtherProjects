import sys
from csv import writer
import os
import pandas as pd
import matplotlib.pyplot as plt

#parser method to extract necessary fields from raw text file
def logParser(dirname, logName):
    input_directory_path = f'logfiles/{dirname}'
    output_directory_path = f'parsed'
    all_in_filenames = os.listdir(input_directory_path)

    for file in all_in_filenames:
        if logName in file:
            fileName = file.split(".")[0]
            #Convert the text file into an array of lines
            with open(f'{input_directory_path}/{file}', encoding="utf8", errors='ignore') as textFile:
                textList = []
                for line in textFile:
                    textList.append(line.strip())

            with open(f'{output_directory_path}/{fileName}_parsed.csv', 'w', newline='') as write_obj:
                csv_writer = writer(write_obj)

                csv_writer.writerow(["Episode", "Reward"])

                #extract relevant elements from the log
                for i in range(0, len(textList)):
                    try:
                        if "cumulative reward:" in textList[i]:
                            episode = textList[i].split("Episode")[1].split(" ")[1]
                            reward = textList[i].split("Episode")[1].split(" ")[4]

                            csv_writer.writerow([episode, reward])
                    except:
                        print("Error printing episode line")
                        continue

def plotter(logname):
    fileName = logname.split(".")[0] + "_parsed.csv"

    input_directory_path = f'parsed'

    parsedData = pd.read_csv(f'{input_directory_path}/{fileName}')
    episode = parsedData["Episode"]
    reward = parsedData["Reward"]

    fig, ax1 = plt.subplots()
    fig.set_size_inches(10, 10)

    for x_val, y_val in zip(episode, reward):
        plt.vlines(x_val, 0, y_val, colors='blue') 

    ax1.set_xlabel('Episode', fontsize=18)
    ax1.set_ylabel('Cumulative Reward', fontsize=18)
    ax1.set_xlim(0,200)
    ax1.set_title("Obstacle DDQN Cumulative Reward", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(f'{input_directory_path}/{logname.split(".")[0]}.png')

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Run with: "python3 rosbagParser.py dirname logname"')
    else:       
        dirname = sys.argv[1]
        logname = sys.argv[2]

        logParser(dirname, logname)
        plotter(logname)