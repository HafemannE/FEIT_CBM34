pip3 install -U sphinx
pip3 install sphinx_rtd_them
sphinx-build sphinx_builder/ docs/
python3 -m  sphinx.cmd.build sphinx_builder/ docs/
Add .nojekyll