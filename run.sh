clear
if gcc neural-network.c -o neural-network -lm; then
./neural-network
python plot_data.py
eog training.png &
else
echo "\n\n Compilation Failed - file not ran";
fi
