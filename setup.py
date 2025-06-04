import numpy.distutils.core

def parse_requirements(path):
    reqs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith('#'):
                reqs.append(line)
    return reqs

if __name__ == "__main__":
    numpy.distutils.core.setup(
        name="pyvar",
        version="0.0.2",
        platforms="linux",
        packages=["pyvar"],
        package_data={"pyvar": ["svar.f90"]},
        test_suite="nose.collector",
        tests_require=["nose"],
        install_requires=parse_requirements("requirements.txt"),
    )
