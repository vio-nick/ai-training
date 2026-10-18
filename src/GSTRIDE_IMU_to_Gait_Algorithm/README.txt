GSTRIDE project has been developed with Python 3.10.4

1) Install Pyhton 3
	
	Install Python 3 Windows:
		a) Go to:
   			https://www.python.org/downloads/windows/
		b) Download last stable release
		c) Execute file downloaded 
		d) During the installation, you will see a "Setup" window. Make sure to select:
			"Add Python 3.X to PATH" or "Add Python to your environment variables"
		e) Click on "Install Now"

	Install Python 3 macOS:
		a) Go to:
   			https://www.python.org/downloads/macos/
		b) Download last stable release
		c) Execute file downloaded

2) Make sure that python is installed

	Windows:
		a) Open cmd in Windows 
		b) Type in cmd: "python --version"
			Command-line should print "Python 3.X"

	macOS:
		a) Open terminal in macOS 
		b) Write in terminal/cmd: "python3 --version"
			Command-line should print "Python 3.X"

3) Install PIP
	
	Install PIP Windows:
		a) Launch a command prompt if it isn't already open(cmd).
		b) Then, run the following command to download the get-pip.py file:
			"curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py"
		c) Type in cmd: "python get-pip.py
		d) Verify installation, type: "pip help"
			If PIP has been installed, you should see the location of the software package and a list of commands you can use with pip.

	Install PIP macOS:	
		a) Launch a Terminal if it isn't already open.
		b) Then, run the following command to download the get-pip.py file:
			"curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py"
		c) Type in terminal: "python3 get-pip.py"
		d) Verify installation, type: "pip --version"
			The output shows which pip version you have on your compute.

4) Install libraries needed to execute main.py:
	a) Launch a Terminal if it isn't already open.
	b) Type and run: "pip install numpy==1.24.3"
	c) Type and run: "pip install matplotlib==3.7.1"
	c) Type and run: "pip install scipy==1.10.1"

5) Download GSTRIDE_database

6) Execute main.py

	Execute main.py Windows:
		a) Open cmd if it isn't already open.
		b) Go to the directory where Gait_analysis_w_Pyhton has been located:
			"cd UserRoot/GSTRIDE_database/Gait_analysis_w_Python"
			For example: "cd Downloads/GSTRIDE_database/Gait_analysis_w_Python"
		c) Execute main.py:
			"python main.py"
	
	Execute main.py macOS:
		a) Open terminal if it isn't already open.
		b) Go to the directory where Gait_analysis_w_Pyhton has been located:
			"cd UserRoot/GSTRIDE_database/Gait_analysis_w_Python"
			For example: "cd Downloads/GSTRIDE_database/Gait_analysis_w_Python"
		c) Execute main.py:
			"python3 main.py"

	
