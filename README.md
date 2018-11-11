# preprocessUCF101
Download the UCF101 data from http://crcv.ucf.edu/data/UCF101/UCF101.rar

Download the UCF101 train-test partation from http://crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip

Unzip UCF101 into ./UCF101/UCF-101/

Unzip label into ./UCF101/ucfTrainTestlist/

Run the python preprocessUCF101.py

Request:

keras

cv2

The folder will be like:

./

--UCF101

-- --ucfTrainTestlist

-- --UCF-101

-- --processeddata

-- -- --part1

-- -- -- --training

-- -- -- --testing

-- -- --part2

-- -- -- --training

-- -- -- --testing

-- -- --part3

-- -- -- --training

-- -- -- --testing
