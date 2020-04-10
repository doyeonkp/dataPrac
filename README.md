Init seting
=============

Anaconda
-------------

## Anaconda Download
1. Download Anaconda from https://www.anaconda.com/distribution/
2. Run Anaconda Powershell Prompt with 'Run as administrator'
3. In the Anaconda shell,  
   <pre> a. check Anaconda's version  
        <code>conda --version</code>
    b. if it needs any update
        <code>conda update conda</code>
    c. make a virtual environment for your project
        <code>conda create --name project's_name installation_package_name</code>
        or
        <code>conda create -n project's_name installation_package_name</code>
        ex) <code>conda create -n practice python=3.5</code>
    d. ativate your project environment
        ex) <code>activate practice</code>
    e. When you need to deactivate your virtual environment 
        ex) <code>deactivate practice</code>
    f. when you want to check virtual environments list
        <code>conda info --envs</code>
    g. install the packages which you need
        <code>conda install package's_name</code>
        [careful] when you install the package, you should activate the virtual environment.
        ex) <code>conda install pandas matplotlib scipy scikit-learn nltk</code>
    h. tensorflow and keras package install
        <code>conda install -c conda-forge tensorflow keras</code>
    i. MatplotLib
        mac precondition: <code>brew install pkg-config</code>
        <code>pip install matplotlib</code>
        <code>pip install git+git://github.com/matplotlib/matplotlib.git</code>
        TIP! if this kind of error show up:
        <code>Matplotlib is building the font cache using fc-list.This may take a moment.</code>
        => <code>rm -rf ~/.cache/fontconfig</code>
        => then try it but still not work
        => <code>import matplotlib as mpl
           print (mpl.get_cachedir ())</code>
        => <code>rm -rf path</code>
            => path is where 'print (mpl.get_cachedir ())'
    j. seaborn
        => <code>pip install seaborn</code>
        => <code>conda install seaborn</code>
        => <code>pip install git+https://github.com/mwaskom/seaborn.git</code>

</pre>
4. install jupyter
   <code>pip install jupyter</code>
   
   
5. run notebook
    <code>jupyter notebook</code>
