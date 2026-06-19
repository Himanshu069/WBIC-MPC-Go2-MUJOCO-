with open("unitree_go2/go2.xml", "r") as f:
    content = f.read()

import re
# Find all body and site names
bodies = re.findall(r'<body\s+name="([^"]+)"', content)
sites  = re.findall(r'<site\s+name="([^"]+)"', content)

print("Bodies:", bodies)
print("Sites:", sites)