'''
Used to read from a file to generate source data.
'''
import numpy as np
import openpyxl
import re
import os

# Add various split characters here
re_split = re.compile(r'[\s\,\;]+')


def readTxT(filename):
    print("Reading {}".format(filename))
    with open(filename, 'r') as f:
        X, y = [], []
        for line in f.readlines():
            #print(type(line), len(line.strip()))
            a = [float(x) for x in re_split.split(line) if x.isdigit()]
            if np.shape(a)[0] < 2:             # shape: row * col, but for one-dimensional list, use [0]
                print('Error dimension.')
            else:
                X.append(float(a[0]))
                y.append(float(a[1]))
        return X, y


# Read excel data using openpyxl
def readExcel(filename):
    if os.path.splitext(filename)[1] != '.xlsx':
        print('The file type is wrong.')
    else:
        print('Reading...')
        wb = openpyxl.load_workbook(filename, 'r')
        #sheetNames = wb.get_sheet_names()
        sheet = wb.get_sheet_by_name(wb.get_sheet_names()[0])
        rowMax = sheet.max_row
        startIdx = 'A' + str(1)
        endIdx = 'A' + str(rowMax)
        X, y = [], []
        print('StartIdx = ', startIdx)
        print('EndIdx = ', endIdx)
        for cellObj in sheet[startIdx : endIdx]:
            for cells in cellObj:
                X.append(cells.value)
        startIdx = 'B' + str(1)
        endIdx = 'B' + str(rowMax)
        for cellObj in sheet[startIdx : endIdx]:
            for cells in cellObj:
                y.append(cells.value)
        return X, y


def readData(filename):
    if not os.path.exists(filename):
        print("File {} doesn't exist.".format(filename))
    else:
        if os.path.splitext(filename)[1] == '.xlsx':
            X, y = readExcel(filename)
        else:
            X, y = readTxT(filename)

        return X, y


