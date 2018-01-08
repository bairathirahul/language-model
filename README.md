# Execution
To execute the program, run following command in the command prompt
`py bairathi_assignment2.py > ouptput.txt`

The command will save the output of program in a file "output.txt". The file will contain calculated probabilities of words of
* English Dataset with English Language Model
* French Dataset with English Language Model
* Italian Dataset with Italian Language Model
* Spanish Dataset with Italian Language Model

The function 'perform_experiment' is the main entry point of the function. It takes four arguments:
* modelFile: The fileid of file from 'UDHR' package on which Language Model must be trained
* modelLanguage: The name of the language of modelFile. It is only used for the output purpose
* dataFile:  The fileid of file from 'UDHR' package from which words must be read and tested on the Language Model
* dataLanguage: The name of the language of dataFile. It is only used for the output purpose.

To test the model on a different language, e.g. Deutsche, call 'perform_experiment' function as follows:
`perform_experiment('German_Deutsch-Latin1', 'German', 'German_Deutsch-Latin1', 'German')`