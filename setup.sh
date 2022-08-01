#it's somewhat important that you run this script from the pyCudaPacking
#directory

pyDiffusionPath=${PWD}

#first we need to ask where you want to put pybind11
while true; do
    read -rep $'Please input the path where you would like pybind11. Leave blank to put in the same directory as pyDiffusionPath.\n' pyBindPath
    if [ "$pyBindPath" = "" ]; then
	pyBindPath=$(echo $pyDiffusionPath | sed "s|/pyDiffusionPath||")
	echo $pyBindPath
    fi
    break
done

#then np_quad:
while true; do
    read -rep $'Please input the path where you would like np_quad. Leave blank to put in the same directory as pyDiffusionPath.\n' npQuadPath
    if [ "$npQuadPath" = "" ]; then
	npQuadPath=$(echo $pyDiffusionPath | sed "s|/pyDiffusionPath||")
	echo $npQuadPath
    fi
    break
done

cd $pyBindPath
# clone the standard version of pybind11
git clone https://github.com/pybind/pybind11.git

cd $npQuadPath
# clone the standard version of np_quad
git clone git@github.com:SimonsGlass/numpy_quad.git
cd numpy_quad/
echo "Now setup your numpy_quad"
python3 setup.py develop --user

# compile the C++ library
cd ../src/
./compile.sh

# now export pyDiffusion to python path
dirPath="$pyDiffusionPath"
exportString='export PYTHONPATH=$PYTHONPATH:'
echo "$exportString$dirPath" >> ~/.bashrc