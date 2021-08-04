"""
Based on:
https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-presigned-urls.html
"""

import os
import logging
import boto3
from botocore.exceptions import ClientError
from datetime import datetime as dt
import json

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def handler(event, context):
    try:
        # Create a unique key for user-uploaded video
        video_id = dt.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        video_type = event["headers"]["content-type"]
        logger.info(f"Upload video type: {video_type}")
        if "quicktime" in video_type:
            video_ext = "mov"
        elif "x-msvideo" in video_type:
            video_ext = "avi"
        else:
            video_ext = "mp4"
        video_key = f"{video_id}.{video_ext}"

        # Get bucket name associated with this lambda function
        region = os.environ["AWS_REGION"]
        bucket = os.environ["UploadBucket"]

        # Get presigned URL for uploading the video
        logger.info(
            f"Getting a presigned URL for object {video_key} in bucket: {bucket}"
        )
        response_url = create_presigned_url(region, bucket, video_key, video_type)
        if not response_url:
            raise ("Failed to get a presigned URL")

        logger.info(f"Presigned URL: {response_url}")
        response_json = json.dumps({"uploadURL": response_url, "Key": video_key})
        return response_json

    except Exception as e:
        logger.exceptions(e)
        logger.error(f"Error getting URL for object {video_key} from bucket {bucket}")
        raise e


def create_presigned_url(
    region_name, bucket_name, object_name, object_type, expiration=3600
):
    """Generate a presigned URL to PUT an S3 object

    :param bucket_name: string
    :param object_name: string
    :param object_type: string
    :param expiration: Time in seconds for the presigned URL to remain valid
    :return: Presigned URL as string. If error, returns None.
    """

    # Generate a presigned URL for the S3 object
    s3_client = boto3.client("s3", region_name=region_name)
    params = {
        "Bucket": bucket_name,
        "Key": object_name,
        "ContentType": object_type,
    }
    logger.info(f"Params: {params}")
    try:
        response = s3_client.generate_presigned_url(
            "put_object",
            Params=params,
            ExpiresIn=expiration,
            HttpMethod="PUT",
        )
    except ClientError as e:
        logger.exceptions(e)
        return None

    # The response contains the presigned URL
    return response
