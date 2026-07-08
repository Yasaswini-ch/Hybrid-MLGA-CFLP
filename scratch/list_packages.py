import pkg_resources
installed_packages = [d.project_name for d in pkg_resources.working_set]
print(sorted(installed_packages))
