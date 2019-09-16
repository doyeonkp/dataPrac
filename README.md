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
</pre>
4. install jupyter
   <code>pip install jupyter</code>
   

ㄴㄻㄻㄴㄹ


5. run notebook
    <code>jupyter notebook</code>
