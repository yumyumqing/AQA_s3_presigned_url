import os
import logging
import boto3
from botocore.exceptions import ClientError
from datetime import datetime as dt
import json

logger = logging.getLogger()
logger.setLevel(logging.INFO)

TABLE_NAME = "aqa_scores"
VIDEO_NAME_KEY = 'videoName'
VIDEO_SCORE_KEY = 'videoScore'

def handler(event, context):
    try:
        logger.info(event)
        raw_query_str = event["rawQueryString"]
        if "=" in raw_query_str:
            video_key = raw_query_str.split("=")[-1]
        else:
            logger.error("Something wrong with the HTTP request, no valid query string available.")
            return None
        
        # Get region name associated with this lambda function
        region = os.environ["AWS_REGION"]
        
        video_score = fetch_db_result(region, video_key)
        return video_score

    except Exception as e:
        logger.exceptions(e)
        raise e


def fetch_db_result(region_name, video_key):
    """Fetch item from DynamoDB of a certain video key

    :param region_name: string
    :param video_key: string
    :return: video_score of DB item with the video_key. If not present, returns None.
    """
    db_client = boto3.client("dynamodb", region_name=region_name)
    logger.info(f"Getting score for [{video_key}] from table:{TABLE_NAME}...")
    try:
        response = db_client.get_item(
            TableName = TABLE_NAME,
            Key = {VIDEO_NAME_KEY: { "S": video_key }},
            ProjectionExpression= VIDEO_SCORE_KEY,
        )
        logger.info(response)
        if "Item" in response:
            score = response["Item"]["videoScore"]["N"]
            logger.info(f"Score for video [{video_key}]: {score}")
        else:
            score = None
            logger.info(f"Score for video [{video_key}] is not available")
        return score
    except ClientError as e:
        logger.exceptions(e)
        return None