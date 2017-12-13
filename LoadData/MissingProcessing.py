from LoadData.DataRecord import *
#generate data ID without position-label-missing

def filterMissingData():
    dataR=DataRecord()
    dataR.countMissing()
    for key in dataR.positionData.keys():
        with open("../data/ProcessedData/"+key+".csv","w",newline="") as f:

            writer=csv.writer(f)

            writer.writerow(["imageID","positionX","positionY"])

            labels=dataR.positionData[key]
            for ID in range(1,dataR.Size+1):
                if ID in labels["maintainID"]:
                    #print(labels[ID])
                    x,y=labels[ID]
                    writer.writerow([ID,x,y])

    with open("../data/ProcessedData/face_image.csv", "w", newline="")as f:
        writer = csv.writer(f)
        writer.writerow(["ImageID", "Image"])
        for ID in range(1,dataR.Size+1):
            writer.writerow([ID, " ".join(map(str,dataR.face_image[ID]))])

#test
if __name__=="__main__":
    filterMissingData()

