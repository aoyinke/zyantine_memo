import requests
import json
import time


class StreamingTest:
    """流式输出测试类"""
    
    def __init__(self):
        self.base_url = "http://localhost:8001/v1/chat/completions"
        self.model = "zyantine-v1"
    
    def send_request(self, payload, stream=False) -> requests.Response:
        """发送通用请求"""
        try:
            response = requests.post(
                self.base_url,
                json=payload,
                stream=stream,
                timeout=30
            )
            return response
        except requests.exceptions.RequestException as e:
            print(f"请求失败: {e}")
            raise
    
    def test_streaming_output(self) -> bool:
        """测试流式输出功能"""
        print("测试流式输出功能...")
        print("=" * 60)
        
        # 测试用的请求体
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": "请详细介绍一下你自己，包括你的核心功能和特点。"}
            ],
            "temperature": 0.7,
            "max_tokens": 1000,
            "stream": True  # 启用流式输出
        }
        
        try:
            response = self.send_request(payload, stream=True)
            
            print(f"响应状态码: {response.status_code}")
            print(f"响应头类型: {response.headers.get('Content-Type')}")
            print("=" * 60)
            print("流式响应内容:")
            print("=" * 60)
            
            full_response = ""
            start_time = time.time()
            chunk_count = 0
            
            # 处理流式响应
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    
                    if line_str == "data: [DONE]":
                        break
                    
                    if line_str.startswith("data: "):
                        try:
                            data = json.loads(line_str[6:])
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            
                            if content:
                                print(content, end='', flush=True)
                                full_response += content
                                chunk_count += 1
                        except json.JSONDecodeError as e:
                            print(f"\nJSON解析错误: {e}")
                        except Exception as e:
                            print(f"\n处理响应时出错: {e}")
            
            end_time = time.time()
            
            print("\n" + "=" * 60)
            print("流式输出测试结果:")
            print("=" * 60)
            print(f"总响应长度: {len(full_response)} 字符")
            print(f"收到的响应片段数: {chunk_count}")
            print(f"总响应时间: {end_time - start_time:.2f} 秒")
            print(f"是否成功接收到完整响应: {len(full_response) > 0}")
            
            return True
            
        except requests.exceptions.RequestException:
            return False
        except KeyboardInterrupt:
            print("\n测试被中断")
            return False
    
    def test_non_streaming_output(self) -> bool:
        """测试非流式输出功能"""
        print("\n\n测试非流式输出功能...")
        print("=" * 60)
        
        # 测试用的请求体
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": "请简单介绍一下你自己。"}
            ],
            "temperature": 0.7,
            "max_tokens": 200,
            "stream": False  # 禁用流式输出
        }
        
        try:
            start_time = time.time()
            response = self.send_request(payload, stream=False)
            end_time = time.time()
            
            print(f"响应状态码: {response.status_code}")
            print(f"响应头类型: {response.headers.get('Content-Type')}")
            print("=" * 60)
            
            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                
                print("非流式响应内容:")
                print(content)
                
                print("=" * 60)
                print(f"总响应长度: {len(content)} 字符")
                print(f"总响应时间: {end_time - start_time:.2f} 秒")
            
            return True
            
        except requests.exceptions.RequestException:
            return False


def main():
    """主函数"""
    print("自衍体AI流式输出功能测试")
    print("=" * 60)
    
    test = StreamingTest()
    
    # 测试流式输出
    streaming_success = test.test_streaming_output()
    
    # 测试非流式输出
    non_streaming_success = test.test_non_streaming_output()
    
    print("\n" + "=" * 60)
    print("测试总结:")
    print("=" * 60)
    print(f"流式输出功能: {'✅ 成功' if streaming_success else '❌ 失败'}")
    print(f"非流式输出功能: {'✅ 成功' if non_streaming_success else '❌ 失败'}")
    
    if streaming_success:
        print("\n🎉 流式输出功能测试成功！")
    else:
        print("\n❌ 流式输出功能测试失败！")


if __name__ == "__main__":
    main()
