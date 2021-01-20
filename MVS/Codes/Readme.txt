		*Files Detail*

The MVS codes contain a Python file and a subfolder that has the prerequisite (model configuration) of the running code. The trained model is removed from "Models" folder because of space constraints.

"Summary_generation.py" contains the code for summarization of multi-view videos. The same code is used over each Raspberry Pi device to extract keyframes from the corresponding input video. This file has a function call for object detection which returns the class ids by acquiring the current frame as an input.



		*How to RUN?*

1. Open the "Summary_generation.py" file.
2. Edit it by changing the directory of keyframes on line-143, depending upon the input provided.
3. After running, a POP-up box will open, select any MVS dataset video, and let the program deal with it.
4. On the blackscreen, information about the mutual information of current processing frames are provided with the number of persons detected.
5. Keyframes will be extracted in the relevant directory.