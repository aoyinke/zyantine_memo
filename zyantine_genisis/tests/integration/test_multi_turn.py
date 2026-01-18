#!/usr/bin/env python3
"""
多轮对话测试脚本 - 测试对话历史缓存修复
"""
import asyncio
import json
import websockets
from datetime import datetime


async def test_multi_turn_conversation():
    """测试多轮对话功能"""
    uri = "ws://localhost:8001/ws"
    client_id = f"test_multi_turn_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    try:
        async with websockets.connect(uri) as websocket:
            print(f"\n{'='*60}")
            print(f"已连接到服务器: {uri}")
            print(f"客户端ID: {client_id}")
            print(f"{'='*60}\n")

            # 发送连接消息
            connect_msg = {
                "type": "connect",
                "client_id": client_id
            }
            await websocket.send(json.dumps(connect_msg, ensure_ascii=False))
            response = await websocket.recv()
            print(f"连接响应: {response}\n")

            # 测试对话列表
            test_conversations = [
                "你好，我叫小明",
                "我今年25岁",
                "你还记得我的名字吗？",
                "我刚才说我多大了？",
                "再见"
            ]

            for i, message in enumerate(test_conversations, 1):
                print(f"\n{'='*60}")
                print(f"对话 {i}/{len(test_conversations)}")
                print(f"{'='*60}")

                # 发送消息
                chat_msg = {
                    "type": "chat",
                    "message": message,
                    "model": "zyantine-v1"
                }

                print(f"\n[发送] {message}")
                await websocket.send(json.dumps(chat_msg, ensure_ascii=False))

                # 接收响应
                response = await websocket.recv()
                response_data = json.loads(response)

                if response_data.get("type") == "response":
                    print(f"[接收] {response_data.get('message')}")
                else:
                    print(f"[错误] {response_data}")

                # 等待一下，模拟真实对话间隔
                await asyncio.sleep(1)

            print(f"\n{'='*60}")
            print(f"多轮对话测试完成")
            print(f"{'='*60}\n")

            # 发送关闭消息
            close_msg = {
                "type": "close"
            }
            await websocket.send(json.dumps(close_msg, ensure_ascii=False))

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


async def test_fitness_test_conversation():
    """测试体测相关的多轮对话功能，特别是"记得这件事情"的上下文关联"""
    uri = "ws://localhost:8001/ws"
    client_id = f"test_fitness_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    try:
        async with websockets.connect(uri) as websocket:
            print(f"\n{'='*60}")
            print(f"已连接到服务器: {uri}")
            print(f"客户端ID: {client_id}")
            print(f"{'='*60}\n")

            # 发送连接消息
            connect_msg = {
                "type": "connect",
                "client_id": client_id
            }
            await websocket.send(json.dumps(connect_msg, ensure_ascii=False))
            response = await websocket.recv()
            print(f"连接响应: {response}\n")

            # 体测相关测试对话
            test_conversations = [
                "我今天去做了体测",
                "我的肺活量是4500毫升",
                "50米跑了7.8秒",
                "记得这件事情",
                "我刚才说我的肺活量是多少来着？",
                "我50米跑了多少秒？",
                "我今天去做了什么？",
                "再见"
            ]

            for i, message in enumerate(test_conversations, 1):
                print(f"\n{'='*60}")
                print(f"对话 {i}/{len(test_conversations)}")
                print(f"{'='*60}")

                # 发送消息
                chat_msg = {
                    "type": "chat",
                    "message": message,
                    "model": "zyantine-v1"
                }

                print(f"\n[发送] {message}")
                await websocket.send(json.dumps(chat_msg, ensure_ascii=False))

                # 接收响应
                response = await websocket.recv()
                response_data = json.loads(response)

                if response_data.get("type") == "response":
                    print(f"[接收] {response_data.get('message')}")
                else:
                    print(f"[错误] {response_data}")

                # 等待一下，模拟真实对话间隔
                await asyncio.sleep(1)

            print(f"\n{'='*60}")
            print(f"体测多轮对话测试完成")
            print(f"{'='*60}\n")

            # 发送关闭消息
            close_msg = {
                "type": "close"
            }
            await websocket.send(json.dumps(close_msg, ensure_ascii=False))

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("开始多轮对话测试...")
    asyncio.run(test_multi_turn_conversation())
    
    print("\n" + "="*60)
    print("开始体测相关多轮对话测试...")
    print("="*60)
    asyncio.run(test_fitness_test_conversation())
