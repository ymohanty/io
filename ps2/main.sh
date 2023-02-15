
#!/bin/sh

####################################################################################################################
#                                     Advanced Industrial Organization II -- Pset 2 Replication File
#                                                    Brian Curran & Yashaswi Mohanty
####################################################################################################################

############################# COMMAND-LINE ARGUMENTS ##########################################

# ~~~~~~~~~~~~ Meta variables for control flow ~~~~~~~~~~~~~~~~~~~~
# Step of the analysis
stage=1

# Boolean to indicate whether we are
# runninig a clean build
clean_build=0

while getopts ":hcr:" opt; do
  case ${opt} in
    h )
      echo "usage: ./main.sh [-h] [-r <step>]\n"
      echo "Option                                  Meaning"
      echo "-h                                      Show this message."
      echo "-r <q>                                  Run analysis for question <q> and greater. Default = 1. Range = 1 - 3 ."
      exit
      ;;
    r )
       stage=$OPTARG
       if [ "$stage" -gt 3 ] || [ "$stage" -lt 1 ]; then
            >&2 echo "Error: provide step arguments between 1 and 9 inclusive. ./main.sh -h for help"
            exit 1
       fi
       ;;
    c )
       if [ "$stage" -ne 1 ]; then
            >&2 echo "Error: clean builds must start at the beginning of the analysis. [ -r 1 ]"
            exit 1
       fi
       echo "Are you sure you wish to run a clean build? [y/n]"    
       read clean_build
       if [ "$clean_build"  == "y" ]; then
            clean_build=1
       fi
       ;;
    \? )
      >&2 echo "Invalid option: $OPTARG"
      >&2 echo "usage: ./main.sh [-h] [-r <step>]"
      exit 1
      ;;
    : )
      echo "Invalid option: $OPTARG requries an argument" 1>&2
      exit 1
      ;;
  esac
done
shift $((OPTIND -1))

################################# FUNCTION DEFINITIONS ##########################################

# @Override
# Override echo for Linux operating systems
machine="$(uname -s)"
if [ "$machine" == "Linux" ]; then
    echo () {
        printf "$1\n\n"
    }
fi


# Handle errors from Stata, R and MATLAB subroutines by using regular expressions to 
# search for error messages in the log files.
# We do NOT want to clutter stdout with subroutine output and so all the output is kept in
# conveniently located logs.
handle_error () {
    local error=$(grep $1 "$2")
    if [ ! -z "$error" ]; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            say error
        fi
        echo "That code ran with error(s): ${error}"
        echo "Check logs at: ${2}"
        exit 1
    fi
}


# @Override 
# Just a spinner to tell you we are not done yet!
wait () {
    local pid=$!
    local spinner="-\|/"

    local i=0
    echo "\n"
    while kill -0 $pid &>/dev/null; do
        i=$(( (i+1) %4 ))
        printf "\rWorking...${spinner:$i:1}"
        sleep .1
    done
    printf "\033[2K"
    echo "\rDone"
}

###################################### PROJECT LEVEL GLOBALS ################################################

home="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"


# Check if figures folder exists
if [ ! -d "${home}/figures" ]; then 
    mkdir "${home}/figures"
fi

################################ CLEAR THE OUTPUT-SPACE ###########################
if [ "$clean_build" -eq 1 ]; then
    
    echo "Clearing the output space"
    sleep 1

    # Clean out virtual environment
    rm -r "${home}/io_ps2/"
    
    # Clean out data
    rm "${home}/data/ps2q2.csv/"

    # Clean out exhibits
    rm "${home}/tables/"*
    rm "${home}/figures/"*

    # Clean out logs
    rm ""

    echo "Done"
    sleep 5
    clear
fi

#  ########## PYTHON #########
if [ -x "$(command -v conda)" ]; then
    if [ -d "./io_ps2" ]; then
        echo "\n Activating conda virtual environment...\n"
        source activate ./io_ps2
    else
        echo "Generating conda virtual environment...\n"
        conda-env create --prefix io_ps2 --file=environment.yml 

        echo "\n Activating conda virtual environment...\n"
        source activate ./io_ps2
    fi
else
    echo "Error: please install conda!"
    exit 1
fi



########################## MAIN #############################

# Question 1
if [ "$stage" -eq 1 ]; then

    cd "${home}"
    echo "Question 1: Entry\n"
    echo "Python script: $(pwd)/question_1.py\n"
    echo "Input(s): $(pwd)/data/ps1_ex1.csv\n"
    echo "Output(s): \n"
    echo "EXHIBITS: \n"
    echo "Logs: $(pwd)/logs/question_1.log\n"

    # Make the logs directory if it does not exist
    if [ ! -d "$(pwd)/logs" ]; then
        mkdir logs
    fi

    # Execute
    python question_1.py 2> ./logs/question_1.errlog
    wait
    handle_error "Error" "$(pwd)/logs/question_1.errlog"

    echo "------------------------------------------------------------"

    # Increment analysis step 
    stage=$((stage + 1)) 
    echo "\n\n\n\n\n"
fi

# Question 2
if [ "$stage" -eq 2 ]; then

    cd "${home}"
    echo "Question 2: Entry and heterogeneity\n"
    echo "Python script: $(pwd)/question_2.py\n"
    echo "Input(s): None\n"
    echo "Output(s): $(pwd)/data/ps2q2.csv\n"
    echo "EXHIBITS: See $(pwd)/figures \n"
    echo "Logs: $(pwd)/logs/question_2.log\n"

    # Make the logs directory if it does not exist
    if [ ! -d "$(pwd)/logs" ]; then
        mkdir logs
    fi

    # Execute
    python question_2.py 2> ./logs/question_2.errlog
    wait
    handle_error "Error" "$(pwd)/logs/question_2.errlog"

    echo "------------------------------------------------------------"

    # Increment analysis step 
    stage=$((stage + 1)) 
    echo "\n\n\n\n\n"
fi

# Question 3
if [ "$stage" -eq 3 ]; then

    cd "${home}"
    echo "Question 3: Dynamic discrete choice\n"
    echo "Python script: $(pwd)/question_3.py\n"
    echo "Input(s): No$(pwd)/data/ps2_ex3.csv\n"
    echo "Output(s): None\n"
    echo "EXHIBITS: See $(pwd)/tables \n"
    echo "Logs: $(pwd)/logs/question_3.log\n"

    # Make the logs directory if it does not exist
    if [ ! -d "$(pwd)/logs" ]; then
        mkdir logs
    fi

    # Execute
    python question_2.py 2> ./logs/question_2.errlog
    wait
    handle_error "Error" "$(pwd)/logs/question_3.errlog"

    echo "------------------------------------------------------------"

    # Increment analysis step 
    stage=$((stage + 1)) 
    echo "\n\n\n\n\n"
fi