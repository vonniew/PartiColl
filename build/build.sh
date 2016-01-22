RELEASE_SYM=Release
DEBUG_SYM=Debug

ON_SYM="ON"
OFF_SYM="OFF"

# Check that an argument was given to the build script.
if [ -z $1 ]; then
    echo "ERROR: You must specify either Debug or Release for the build."
    exit -1
fi

# Check that the argument given is either Debug or Release
if [ $1 != $RELEASE_SYM -a $1 != $DEBUG_SYM ]; then
    echo "ERROR: The given argument was neither Debug nor Release."
    exit -1
fi

# Make sure that ON or OFF is specified for Unit testing.
if [ $2 != $ON_SYM -a $2 != $OFF_SYM ]; then
    echo "ERROR: The second argument must be 'ON' or 'OFF' ('ON' to enable unit testing)."
    exit -1
fi

# Build the cmake project in the previous directory.
echo "Building in $1 mode."
cmake .. -DCMAKE_BUILD_TYPE=$1 -DGTEST=$2

# Make sure that the CMake project was built successfully.
if [ $? -ne 0 ]; then
    echo "CMake did not run successfully."
    exit -1
fi

# If "nproc" exists, then run it to figure out how many cores are available for building.
# Otherwise just set the default to 4 cores (arbitrarily chosen).
if type -p nproc; then
    NUM_PROCESSORS=$(($(nproc) + 1))
else
    NUM_PROCESSORS=4 #Defaulting to 4 processors.
fi

echo "Building with $NUM_PROCESSORS cores."

# Build the project.
make -j $NUM_PROCESSORS VERBOSE=1

# Check if the project was built successfully.
if [ $? -ne 0 ]; then
    echo "The project could not be built."
    exit -1
fi

echo "The library was built successfully."
