## 问题分析
从日志和代码分析来看，reply_generator组件未调用API回答用户问题的主要原因是：

1. **API服务初始化失败**：导致reply_generator使用了本地模式（api_service=None）
2. **fact_checker中的_extract_statements方法返回None**：导致在协议审查阶段出现`argument of type 'NoneType' is not iterable`错误
3. **fallback策略执行异常**：当API不可用时，fallback策略可能也执行失败，最终调用了emergency_reply

## 修复方案

### 1. 修复fact_checker.py中的_extract_statements方法
- 确保_extract_statements方法始终返回列表，即使在出现异常时
- 在_build_verification_input方法中添加对statements的空值检查

### 2. 增强API服务初始化的健壮性
- 在service_provider.py的_initialize_services方法中添加更详细的日志
- 确保即使API服务初始化失败，也能提供有用的错误信息
- 增强对API服务客户端的检查

### 3. 改进fallback策略的执行
- 在fallback_strategy.py中添加更多的错误处理
- 确保generate_fallback_reply方法始终返回有效的回复

### 4. 优化reply_generator的错误处理
- 在reply_generator.py中添加对api_service的更详细检查
- 确保即使在各种异常情况下，也能返回合理的回复

## 具体修改点

1. **fact_checker.py**：
   - 修改_extract_statements方法，确保始终返回列表
   - 在_build_verification_input方法中添加对statements的空值检查

2. **service_provider.py**：
   - 增强API服务初始化的日志记录
   - 确保API服务客户端初始化失败时，提供明确的错误信息

3. **reply_generator.py**：
   - 增强对api_service的检查
   - 优化错误处理逻辑

4. **fallback_strategy.py**：
   - 增强错误处理
   - 确保始终返回有效的回复

## 预期效果
- API服务初始化过程更加健壮，提供详细的日志信息
- fact_checker不再因_extract_statements返回None而崩溃
- reply_generator在各种情况下都能返回合理的回复
- 系统整体稳定性提高，减少意外错误