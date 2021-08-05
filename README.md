This repo is based on https://github.com/aws-samples/amazon-s3-presigned-urls-aws-sam

# S3 presigned URLs with SAM, auth and sample frontend

This example application shows how to upload objects to S3 directly from your end-user application using Signed URLs.

# Added

Also added a S3 trigger lambda function that executes whenever an object is created in the S3 bucket.

To learn more about how this application works, see the AWS Compute Blog post: https://aws.amazon.com/blogs/compute/uploading-to-amazon-s3-directly-from-a-web-or-mobile-application/

Important: this application uses various AWS services and there are costs associated with these services after the Free Tier usage - please see the [AWS Pricing page](https://aws.amazon.com/pricing/) for details. You are responsible for any AWS costs incurred. No warranty is implied in this example.

```bash
.
├── README.MD                   <-- This instructions file
├── frontend                    <-- Simple JavaScript application illustrating upload
├── getSignedURL                <-- Source code for the serverless backend to get a presigned URL for upload
├── getPredScore                <-- Source code for the serverless backend to download object from S3 and process
```

## Requirements

* AWS CLI already configured with Administrator permission
* [AWS SAM CLI installed](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-sam-cli-install.html) - minimum version 0.48.
* python3.8

## Installation Instructions

1. [Create an AWS account](https://portal.aws.amazon.com/gp/aws/developer/registration/index.html) if you do not already have one and login.

2. Clone the repo onto your local development machine using `git clone`.

### Installing the application

There are two SAM templates available - one provides an open API, the other uses an authorizer. From the command line, deploy the chosen SAM template:

```
cd .. 
sam deploy --guided
```

When prompted for parameters, enter:
- Stack Name: AQAUploadProcess
- AWS Region: us-west-2
- Accept all other defaults. Except:
```
UploadRequestFunction may not have authorization defined, Is this okay? [y/N]: y
```

This takes several minutes to deploy. At the end of the deployment, note the output values, as you need these later.

- The APIendpoint value is important - it looks like https://ab123345677.execute-api.us-west-2.amazonaws.com.
- **The upload URL is your endpoint** with the /uploads route added - for example: https://ab123345677.execute-api.us-west-2.amazonaws.com/uploads.

### Testing with Postman

1. First, copy the API endpoint from the output of the deployment.
   https://glxlp92kil.execute-api.us-west-2.amazonaws.com/uploads
2. In the Postman interface, paste the API endpoint into the box labeled Enter request URL, select GET, add video as binary raw file.
3. Send
4. (For uploading a file to S3) In the Postman interface, open another tab, paste the presigned URL get from step 3, select PUT, add the same video as binary raw file as in step 2, send

### Testing with the frontend application (THIS IS FROM THE ORIGINAL REPO AT THE TOP, NOT WORKING FOR THIS REPO)

The frontend code is saved in the `frontend` subdirectory. 

1. Before running, you need to set the API Gateway endpoint from the backend deployment on line 29 in the `index.html` file.

2. You cannot run this directly on a local browser, due to way CORS works with localhost. Either [copy the file to an S3 bucket](https://docs.aws.amazon.com/AmazonS3/latest/user-guide/upload-objects.html), or [deploy using AWS Amplify Console](https://aws.amazon.com/amplify/console/).

3. Once the page is loaded from a remote location, upload a JPG file in the front-end and you will see the object in the backend S3 bucket.

## Next steps

The AWS Compute Blog post at the top of this README file contains additional information about this pattern.

If you have any questions, please raise an issue in the GitHub repo.

==============================================

Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.

SPDX-License-Identifier: MIT-0
