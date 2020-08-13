import boto3
def upload_file(localfile, file_name, bucket):
    """
    Function to upload a file to S3 bucket
    """
    print("Inside s3files.py: ", file_name)
    print(" ")
    # object_name = file_name
    s3_client = boto3.client('s3')
    # response = s3_client.upload_file(file_name, bucket, object_name)
    response = s3_client.upload_file(localfile, bucket, file_name)

    return response

# def download_file(subfolder, file_name, bucket):
#     """
#     Function to download a given file from S3 bucket
#     """
#     print("File Name:", file_name)
#     print("Sub Folder:", subfolder)

#     s3 = boto3.resource('s3')
#     if subfolder != '':
#         file_name2 = f"{subfolder}/{file_name}"
#     else:
#         file_name2 = file_name

#     output = f"downloads/{file_name}"
#     s3.Bucket(bucket).download_file(file_name2, output)

#     return output

def download_file(file_name, bucket):
    """
    Function to download a given file from S3 bucket
    """
    print("File Name:", file_name)

    s3 = boto3.resource('s3')

    output = f"downloads/{file_name}"
    s3.Bucket(bucket).download_file(file_name, output)

    return output

def list_files(bucket):
    """
    Function to list files in a given S3 bucket
    """
    s3 = boto3.client('s3')
    contents = []
    for item in s3.list_objects(Bucket=bucket)['Contents']:
        contents.append(item)

    return contents