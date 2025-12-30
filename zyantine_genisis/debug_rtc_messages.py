"""
RTC 消息格式调试脚本
用于诊断 RTC 系统发送到 WebSocket 的消息格式问题
"""
import asyncio
import json
import websockets
from typing import Optional


class RTCMessageDebugger:
    """RTC 消息调试器"""

    def __init__(self, uri: str, client_id: str = "rtc_debug"):
        self.uri = uri
        self.client_id = client_id
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None

    async def connect(self):
        """连接到服务器"""
        try:
            self.websocket = await websockets.connect(self.uri)
            print(f"\n{'='*60}")
            print(f"已连接到服务器: {self.uri}")
            print(f"{'='*60}\n")
            return True
        except Exception as e:
            print(f"连接失败: {e}")
            return False

    async def send_raw(self, data: bytes, description: str = ""):
        """发送原始字节数据"""
        if not self.websocket:
            print("未连接到服务器")
            return None

        print(f"\n{'='*60}")
        print(f"{description}")
        print(f"{'='*60}")

        print(f"\n[客户端] 准备发送原始数据:")
        print(f"  数据类型: {type(data)}")
        print(f"  数据长度: {len(data)}")
        print(f"  数据内容 (bytes): {data}")
        print(f"  数据内容 (repr): {repr(data)}")
        print(f"  数据内容 (hex): {data.hex()}")

        try:
            await self.websocket.send(data)
            print(f"\n[客户端] 原始数据已发送")

            response = await self.websocket.recv()
            print(f"\n[客户端] 收到响应:")
            print(f"  响应类型: {type(response)}")
            print(f"  响应长度: {len(response)}")
            print(f"  响应内容: {repr(response)}")

            try:
                response_data = json.loads(response)
                print(f"\n[客户端] JSON 解析成功:")
                print(f"  响应数据: {response_data}")
                return response_data
            except json.JSONDecodeError as e:
                print(f"\n[客户端] JSON 解析失败: {e}")
                return None

        except Exception as e:
            print(f"\n[客户端] 发送失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def send_text(self, text: str, description: str = ""):
        """发送文本数据"""
        if not self.websocket:
            print("未连接到服务器")
            return None

        print(f"\n{'='*60}")
        print(f"{description}")
        print(f"{'='*60}")

        print(f"\n[客户端] 准备发送文本:")
        print(f"  文本类型: {type(text)}")
        print(f"  文本长度: {len(text)}")
        print(f"  文本内容: {repr(text)}")
        print(f"  文本编码 (UTF-8): {text.encode('utf-8')}")
        print(f"  文本编码 (GBK): {text.encode('gbk', errors='ignore')}")

        try:
            await self.websocket.send(text)
            print(f"\n[客户端] 文本已发送")

            response = await self.websocket.recv()
            print(f"\n[客户端] 收到响应:")
            print(f"  响应类型: {type(response)}")
            print(f"  响应长度: {len(response)}")
            print(f"  响应内容: {repr(response)}")

            try:
                response_data = json.loads(response)
                print(f"\n[客户端] JSON 解析成功:")
                print(f"  响应数据: {response_data}")
                return response_data
            except json.JSONDecodeError as e:
                print(f"\n[客户端] JSON 解析失败: {e}")
                return None

        except Exception as e:
            print(f"\n[客户端] 发送失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def test_standard_json(self):
        """测试标准 JSON 格式"""
        payload = {
            "type": "chat",
            "message": "你好，这是标准 JSON 格式",
            "model": "zyantine-v1"
        }
        json_str = json.dumps(payload, ensure_ascii=False)
        await self.send_text(json_str, "测试 1: 标准 JSON 格式")

    async def test_json_with_ascii(self):
        """测试 JSON 带 ASCII 转义"""
        payload = {
            "type": "chat",
            "message": "你好，这是带 ASCII 转义的 JSON",
            "model": "zyantine-v1"
        }
        json_str = json.dumps(payload, ensure_ascii=True)
        await self.send_text(json_str, "测试 2: JSON 带 ASCII 转义")

    async def test_json_bytes_utf8(self):
        """测试 JSON 字节（UTF-8 编码）"""
        payload = {
            "type": "chat",
            "message": "你好，这是 UTF-8 编码的字节",
            "model": "zyantine-v1"
        }
        json_str = json.dumps(payload, ensure_ascii=False)
        json_bytes = json_str.encode('utf-8')
        await self.send_raw(json_bytes, "测试 3: JSON 字节（UTF-8 编码）")

    async def test_json_bytes_gbk(self):
        """测试 JSON 字节（GBK 编码）"""
        payload = {
            "type": "chat",
            "message": "你好",
            "model": "zyantine-v1"
        }
        json_str = json.dumps(payload, ensure_ascii=False)
        try:
            json_bytes = json_str.encode('gbk')
            await self.send_raw(json_bytes, "测试 4: JSON 字节（GBK 编码）")
        except Exception as e:
            print(f"\n[客户端] GBK 编码失败: {e}")

    async def test_double_encoded(self):
        """测试双重编码（常见问题）"""
        payload = {
            "type": "chat",
            "message": "你好",
            "model": "zyantine-v1"
        }
        json_str = json.dumps(payload, ensure_ascii=False)
        
        # 模拟双重编码问题
        double_encoded = json_str.encode('utf-8').decode('latin-1')
        await self.send_text(double_encoded, "测试 5: 双重编码（UTF-8 -> Latin-1）")

    async def test_url_encoded(self):
        """测试 URL 编码"""
        import urllib.parse
        payload = {
            "type": "chat",
            "message": "你好，这是 URL 编码",
            "model": "zyantine-v1"
        }
        json_str = json.dumps(payload, ensure_ascii=False)
        url_encoded = urllib.parse.quote(json_str)
        await self.send_text(url_encoded, "测试 6: URL 编码")

    async def test_base64_encoded(self):
        """测试 Base64 编码"""
        import base64
        payload = {
            "type": "chat",
            "message": "你好，这是 Base64 编码",
            "model": "zyantine-v1"
        }
        json_str = json.dumps(payload, ensure_ascii=False)
        base64_encoded = base64.b64encode(json_str.encode('utf-8')).decode('ascii')
        await self.send_text(base64_encoded, "测试 7: Base64 编码")

    async def test_malformed_json(self):
        """测试格式错误的 JSON"""
        malformed_json = '{"type": "chat", "message": "你好, "model": "zyantine-v1"}'
        await self.send_text(malformed_json, "测试 8: 格式错误的 JSON")

    async def test_json_without_quotes(self):
        """测试没有引号的 JSON（常见错误）"""
        no_quotes_json = '{type: "chat", message: "你好", model: "zyantine-v1"}'
        await self.send_text(no_quotes_json, "测试 9: 没有引号的 JSON")

    async def test_json_single_quotes(self):
        """测试单引号的 JSON（常见错误）"""
        single_quotes_json = "{'type': 'chat', 'message': '你好', 'model': 'zyantine-v1'}"
        await self.send_text(single_quotes_json, "测试 10: 单引号的 JSON")

    async def test_unicode_escape(self):
        """测试 Unicode 转义"""
        payload = {
            "type": "chat",
            "message": "\u4f60\u597d\uff0c\u8fd9\u662f Unicode \u8f6c\u4e49",
            "model": "zyantine-v1"
        }
        json_str = json.dumps(payload, ensure_ascii=False)
        await self.send_text(json_str, "测试 11: Unicode 转义")

    async def test_raw_text(self):
        """测试纯文本（非 JSON）"""
        await self.send_text("你好，这是纯文本", "测试 12: 纯文本（非 JSON）")

    async def close(self):
        """关闭连接"""
        if self.websocket:
            await self.websocket.close()
            print("\n连接已关闭")


async def run_all_tests(uri: str):
    """运行所有测试"""
    debugger = RTCMessageDebugger(uri, client_id="rtc_debug")

    if not await debugger.connect():
        print("无法连接到服务器，测试终止")
        return

    try:
        await debugger.test_standard_json()
        await asyncio.sleep(1)

        await debugger.test_json_with_ascii()
        await asyncio.sleep(1)

        await debugger.test_json_bytes_utf8()
        await asyncio.sleep(1)

        await debugger.test_json_bytes_gbk()
        await asyncio.sleep(1)

        await debugger.test_double_encoded()
        await asyncio.sleep(1)

        await debugger.test_url_encoded()
        await asyncio.sleep(1)

        await debugger.test_base64_encoded()
        await asyncio.sleep(1)

        await debugger.test_malformed_json()
        await asyncio.sleep(1)

        await debugger.test_json_without_quotes()
        await asyncio.sleep(1)

        await debugger.test_json_single_quotes()
        await asyncio.sleep(1)

        await debugger.test_unicode_escape()
        await asyncio.sleep(1)

        await debugger.test_raw_text()

        print(f"\n{'='*60}")
        print("所有测试完成")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"\n测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await debugger.close()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="RTC 消息格式调试工具")

    parser.add_argument("--uri", "-u", default="ws://localhost:8001/ws", help="WebSocket 服务器地址")
    parser.add_argument("--client-id", "-c", default="rtc_debug", help="客户端 ID")

    args = parser.parse_args()

    uri = f"{args.uri}?client_id={args.client_id}"

    asyncio.run(run_all_tests(uri))


if __name__ == "__main__":
    main()
