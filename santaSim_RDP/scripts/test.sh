#Run python script in the background
python3 Simulation.py -o "TestOutputs" -t 10 &
pidPython=$!

#Kill python if script is killed.
trap cleanup SIGINT

cleanup()
{
    kill $pidPython
    pidJava=$(pgrep java)
    kill $pidJava
}

# While python is running
while kill -0 $pidPython 2>/dev/null; do
    ramusage=$(free | awk '/Mem/{printf("RAM Usage: %.2f\n"), $3/$2*100}'| awk '{print $3}'); echo "Current Memory Usage is: $ramusage%"; sleep 360
done

# Disable the trap on a normal exit.
trap - EXIT