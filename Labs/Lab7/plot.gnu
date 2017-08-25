#!/usr/bin/gnuplot
# To PLOT A SET OF DATA POINTS
# ============================
# Tell gnuplot to write the output into a postscript file
set terminal postscript portrait enhanced

# Tell gnupolot what is the output file name
#set out "myplot_file_name.ps"

# Tell gnuplot to plot your data in the file "mydata.dat"
# with range of 0-100 on both the axes
#plot [0:100] [0:100] 'mydata.dat'

# Now you have the plot in "myplot_file_name.ps"

# TO PLOT A FUNCTION
# ==================
# To define a (linear) function
f(x) = a * x + b

# To set some values for variables
a = 1
b = 1

# To set the title and to label the axes

set xlabel "X axis"
set ylabel "Y axis"

# Tell gnupolot what is the output file name
set out "plot_file.ps"

# Tell gnuplot to plot a function
fit f(x) 'test.dat' using 1:2 via a,b

# Now you have the plot in "myfunction_plot_file_name.ps"