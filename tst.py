import hashlib

text = "Hello World"

# Encode the string to bytes and calculate the MD5 hash
md5_hash = hashlib.md5(text.encode('utf-8')).hexdigest()

print(md5_hash)
# Output: b10a8db164e0754105b7a99be72e3fe5
